import numpy as np
import numpy.linalg as la
import pickle as pkl
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

import torch as th
from torch.autograd import Variable
from VAE import VAE

"""
Notes (from scholarpedia):
    -The SPECTRAL RADIUS of the reservoir weights codetermines:
        (1): (?)
        (2): amount of nonlinear interaction of input components through time 
                (larger spectral radius ==> longer-range interactions)
    -INPUT SCALING codetermines the degree of nonlinearity of the reservoir dynamics. Examples:
        (1): very small input amplitudes ==> reservoir behaves almost like linear medium.
        (2): very large input amplitudes ==> drives the reservoir neurons to the saturation of the
                                              sigmoid, and a binary switching dynamic results.
    -OUTPUT FEEDBACK SCALING determines the extent to which the trained ESN has an autonomous
     generation component.
        (1):      no output feedback: ESN unable to generate predictions for future time steps.
        (2): nonzero output feedbacl: danger of dynamical instability.
    -CONNECTIVITY/SPARSITY of reservoir weight matrix:
        (1) todo
        
"""


class Reservoir(object):
    """
    input_size (K): input signal is K dimensions.
    num_units  (N): reservoir has N units.
    """

    def __init__(self, input_size, num_units, echo_param=0.6, idx=None, activation=np.tanh, 
                    debug=False):
        self.K = input_size
        self.N = num_units
        self.echo_param = echo_param
        self.activation = activation
        self.idx = idx                # <- can assign reservoir a unique ID for debugging
        self.debug = debug

        # input-to-reservoir, reservoir-to-reservoir weights (not yet initialized)
        self.W_in = np.zeros((self.N, self.K))
        self.W_res = np.zeros((self.N, self.N))
        self.state = np.zeros(self.N)            # <- unit states

        # These parameters are initialized upon calling initialize_input_weights()
        # and initialize_reservoir_weights().
        self.spectral_scale = None
        self.W_res_init_strategy = None
        self.input_weights_scale = None
        self.W_in_init_strategy = None

        # helpful information to track
        if self.debug:
            self.signals = [] # <- reservoir states over time during training
            self.num_to_store = 10
        self.ins_init = False; self.res_init = False

    def info(self):
        out = u'Reservoir(N=%d, K=%d, \u03B5=%.2f)\n' % (self.N, self.K, self.echo_param)
        out += 'W_res - spec_scale: %.2f, %s init\n' % (self.spectral_scale, self.W_res_init_strategy)
        out += 'W_in  -      scale: %.2f, %s init' % (self.input_weights_scale, self.W_in_init_strategy)

    def initialize_input_weights(self, strategy='binary', scale=1e-2, offset=0.5):
        self.input_weights_scale = scale
        self.W_in_init_strategy = strategy
        if strategy == 'binary':
            self.W_in = (np.random.rand(self.N, self.K) > 0.5).astype(float)
        elif strategy == 'uniform':
            self.W_in = np.random.rand(self.N, self.K)
        elif strategy == 'gaussian':
            self.W_in = np.random.randn(self.N, self.K)
        else:
            raise ValueError('unknown input weight init strategy %s' % strategy)

        self.W_in -= offset
        self.W_in *= self.input_weights_scale
        self.ins_init = True

    def initialize_reservoir_weights(self, strategy='uniform', spectral_scale=1.0, offset=0.5, 
                                     sparsity=1.0):
        self.spectral_scale = spectral_scale
        self.W_res_init_strategy = strategy
        if strategy == 'binary':
            self.W_res = (np.random.rand(self.N, self.N) > 0.5).astype(float)
        elif strategy == 'uniform':
            self.W_res = np.random.rand(self.N, self.N)
        elif strategy == 'gaussian':
            self.W_res = np.random.randn(self.N, self.N)
        else:
            raise ValueError('unknown res. weight init strategy %s' % strategy)

        # apply the sparsity
        sparsity_matrix = (np.random.rand(self.N, self.N) < sparsity).astype(float)
        self.W_res *= sparsity_matrix

        self.W_res -= offset
        self.W_res /= max(abs(la.eig(self.W_res)[0]))
        self.W_res *= self.spectral_scale
        self.res_init = True

    def forward(self, u_n):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        u_n: K-dimensional input vector
        """
        u_n = u_n.squeeze()
        assert (self.K == 1 and u_n.shape == ()) or u_n.shape[0] == self.W_in.shape[1], \
            "u(n): %s.  W_res: %s (ID=%d)" % (u_n.shape, self.W_res.shape, self.idx)
        assert self.ins_init, "Res. input weights not yet initialized (ID=%d)." % self.idx
        assert self.res_init, "Res. recurrent weights not yet initialized (ID=%d)." % self.idx

        in_to_res = np.dot(self.W_in, u_n).squeeze()
        res_to_res = np.dot(self.state.reshape(1, -1), self.W_res)

        # Equation (1) in "Formalism and Theory" of Scholarpedia page
        self.state = (1. - self.echo_param) * self.state + self.echo_param * self.activation(in_to_res + res_to_res)
        if self.debug:
            self.signals.append(self.state[:self.num_to_store].tolist())

        return self.state.squeeze()

class LayeredESN(object):
    """
    (ABSTRACT CLASS)
    Abstract class for echo state networks (ESNs).
    ESN is this class, but with num_reservoirs=1.

    --------------------------------------------------------------------------------
    |       Argument      |       dtype        |  Description                      |
    -------------------------------------------------------------------------------|
    |         input_size  |  int               | Dimensionality of input signal.   |
    |         output_size |  int               | Dimensionality of output signal.  |
    |      num_reservoirs |  int               | Number of reservoirs.             |
    |     reservoir_sizes |  int   OR [int]    | Size of all reservoirs, OR a list |
    |                     |                    |   containing each of their sizes. |
    |         echo_params |  float OR [float]  | Echo parameter of all reservoirs, |
    |                     |                    |   or list with each echo param.   |
    |   output_activation |  function          | Output layer activation.          |
    | init_echo_timesteps |  int               | Number of timesteps to 'warm up'  |
    |                     |                    |   model.                          |
    |         regulariser |  float             | Regularization strength (lambda). |
    |               debug |  bool              | Debug information.                |
    -------------------------------------------------------------------------------


    """

    def __init__(self, input_size, output_size, num_reservoirs, reservoir_sizes=None,
                 echo_params=0.6, output_activation=None, init_echo_timesteps=100,
                 regulariser=1e-8, activation=np.tanh, debug=False):
        # IMPLEMENTATION STUFF ===================================================
        if input_size != output_size:
            raise NotImplementedError('num input dims must equal num output dims.')
        if output_activation is not None:
            raise NotImplementedError('non-identity output activations not implemented.')
        # ========================================================================
        self.K = input_size
        self.L = output_size
        self.num_reservoirs = num_reservoirs
        self.reservoir_sizes = reservoir_sizes
        self.reservoirs = []

        if reservoir_sizes is None:
            reservoir_sizes = [int(np.ceil(1000. / num_reservoirs))]*num_reservoirs
        elif type(reservoir_sizes) not in [list, np.ndarray]:
            reservoir_sizes = [reservoir_sizes]*num_reservoirs
        if type(echo_params) not in [list, np.ndarray]:
            echo_params = [echo_params]*num_reservoirs
            
        assert len(reservoir_sizes) == self.num_reservoirs

        self.debug = debug

        # initialize reservoirs
        self.__reservoir_input_size_rule__(reservoir_sizes, echo_params, activation)

        self.regulariser = regulariser
        self.init_echo_timesteps = init_echo_timesteps

        if output_activation is None:
            def iden(x): return x
            output_activation = iden
        self.output_activation = output_activation

        self.N = sum(reservoir_sizes)
        self.W_out = np.ones((self.L, self.K+self.N))

    def initialize_input_weights(self, strategies='binary', scales=1e-2, offsets=0.5):
        if type(strategies) not in [list, np.ndarray]:
            strategies = [strategies]*self.num_reservoirs
        if type(scales) not in [list, np.ndarray]:
            scales = [scales]*self.num_reservoirs
        if type(offsets) not in [list, np.ndarray]:
            offsets = [offsets]*self.num_reservoirs

        for i, (strat, scale) in enumerate(zip(strategies, scales)):
            self.reservoirs[i].initialize_input_weights(strategy=strat, scale=scale)

    def initialize_reservoir_weights(self, strategies='uniform', spectral_scales=1.0, offsets=0.5, sparsity=1.0):
        if type(strategies) not in [list, np.ndarray]:
            strategies = [strategies]*self.num_reservoirs
        if type(spectral_scales) not in [list, np.ndarray]:
            spectral_scales = [spectral_scales]*self.num_reservoirs
        if type(offsets) not in [list, np.ndarray]:
            offsets = [offsets]*self.num_reservoirs

        for i, (strat, scale, offset) in enumerate(zip(strategies, spectral_scales, offsets)):
            self.reservoirs[i].initialize_reservoir_weights(strat, scale, offset, sparsity=sparsity)

    @abstractmethod
    def __forward_routing_rule__(self, u_n):
        """
        Abstract function describing how the inputs are passed from layer to layer.
        It should take the input signal as input, and return an array containing
          the concatenated states of all reservoirs.

        This base version returns an empty array, which will cause the network to
          do linear regression on the input signal only.
        """
        return np.array(0)

    @abstractmethod
    def __reservoir_input_size_rule__(self, *args):
        pass

    def forward(self, u_n, calculate_output=True):
        """
        Forward-propagate signal through network.
        If calculate_output = True: returns output signal, y_n.
                              else: returns updated system states, x_n.
        """
        u_n = u_n.squeeze()
        assert (self.K == 1 and u_n.shape == ()) or len(u_n) == self.K

        x_n = self.__forward_routing_rule__(u_n)

        if calculate_output:
            z_n = np.append(x_n, u_n)
            output = self.output_activation(np.dot(self.W_out, z_n))
            return output.squeeze()
        else:
            return x_n

    def train(self, X, y):
        assert X.shape[1] == self.K, "Training data has unexpected dimensionality (%s). K = %d." % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)

        # First, run a few inputs into the reservoir to get it echoing
        initial_data = X[:self.init_echo_timesteps]
        for u_n in initial_data:
            _ = self.forward(u_n, calculate_output=False)

        # Now train the output weights
        X_train = X[self.init_echo_timesteps:]
        D = y[self.init_echo_timesteps:]
        S = np.zeros((X_train.shape[0], self.N+self.K))
        for n, u_n in enumerate(X_train):
            x_n = self.forward(u_n, calculate_output=False)
            z_n = np.append(x_n, u_n)
            S[n, :] = z_n

        # Solve linear system
        T1 = np.dot(D.T, S)
        T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(self.K + self.N))
        self.W_out = np.dot(T1, T2)
        
    def reset_reservoir_states(self):
        for reservoir in self.reservoirs:
            reservoir.state *= 0.

    @property
    def input_size(self):
        return self.K
    
    @property
    def output_size(self):
        return self.L
    
    @property
    def total_num_reservoir_units(self):
        return self.N
    
    @property
    def num_predictor_variables(self):
        return self.N


class ESN(LayeredESN):
    """
    Sub-class of LayeredESN, but this one still allowed to take non-plural argument names, 
        e.g. 'echo_param' instead of 'echo_params', etc.
    """

    def __init__(self, input_size, output_size, reservoir_sizes=None, 
                 echo_params=None, output_activation=None, init_echo_timesteps=100, 
                 regulariser=1e-8, activation=np.tanh, debug=False, **kwargs):
        
        # Messy code allowing ESN init arguments to have multiple names ==========
        if 'num_reservoirs' in kwargs.keys():
            warnings.warn('"num_reservoirs" argument passed to basic ESN. Ignoring.')
        reservoir_sizes = self._handle_arg('reservoir_size', reservoir_sizes, 1000, kwargs)
        echo_params = self._handle_arg('echo_param', echo_params, 0.85, kwargs)
        # ========================================================================

        super(ESN, self).__init__(
            input_size, output_size, num_reservoirs=1, reservoir_sizes=reservoir_sizes,
            echo_params=echo_params, output_activation=output_activation, 
            init_echo_timesteps=init_echo_timesteps, regulariser=regulariser,
            activation=activation, debug=debug
        )

    def initialize_input_weights(self, strategies='binary', scales=1e-2, offsets=0.5, **kwargs):
        strategies = self._handle_arg('strategy', strategies, 'binary', kwargs)
        scales = self._handle_arg('scale', scales, 1e-2, kwargs)
        offsets = self._handle_arg('offset', offsets, 0.5, kwargs)
        super(ESN, self).initialize_input_weights(strategies, scales, offsets)

    def initialize_reservoir_weights(self, strategies='uniform', spectral_scales=1.0, offsets=0.5, sparsity=1.0, **kwargs):
        strategies = self._handle_arg('strategy', strategies, 'uniform', kwargs)
        scales = self._handle_arg('spectral_scale', spectral_scales, 1.0, kwargs)
        offsets = self._handle_arg('offset', offsets, 0.5, kwargs)
        super(ESN, self).initialize_reservoir_weights(strategies, scales, offsets, sparsity)

    def _handle_arg(self, arg_name, arg_plural, default, kwargs):
        out = None
        if arg_name in kwargs.keys():
            out = kwargs[arg_name]
        elif arg_plural is not None:
            out = arg_plural
        else:
            out = default

        assert type(out) not in [list, np.ndarray] or len(out) == 1, "Basic ESN received multiple %s args." % arg_name
        return out

    def __forward_routing_rule__(self, u_n):
        return self.reservoirs[0].forward(u_n)
    
    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        self.reservoirs.append(
            Reservoir(self.K, reservoir_sizes[0], echo_params[0], idx=0, activation=activation, debug=self.debug)
        )

    
class DHESN(LayeredESN):
    """
    (int or [int]) dim_reduce: dimensionalities of encoders.
    """

    def __init__(self, *args, **kwargs):
        # Encoder dimensionalities =========================================
        if 'dim_reduce' not in kwargs.keys():
            self.dim_reduce = 100
        else:
            self.dim_reduce = kwargs['dim_reduce']
            del kwargs['dim_reduce']
            
        # ==================================================================
        # Encoder type [PCA, AE, ELM, ...] =================================
        if 'encoder_type' not in kwargs.keys():
            self.encoder_type = 'PCA'
        else:
            self.encoder_type = kwargs['encoder_type']
            del kwargs['encoder_type']
        
        super(DHESN, self).__init__(*args, **kwargs)
        
        # ==================================================================
        # Initialize encoders ==============================================
        self.encoders = []
        if self.encoder_type == 'PCA':
            for j in range(self.num_reservoirs-1):
                # self.encoders.append(PCA(n_components=self.reservoirs[j-1].N))
                self.encoders.append(PCA(n_components=self.dim_reduce))
        elif self.encoder_type == 'VAE':
            for j in range(self.num_reservoirs-1):
                self.encoders.append(VAE(input_size=self.reservoir_sizes[j-1], hidden_size=150, latent_variable_size=self.dim_reduce,
                                            epochs=10, batch_size=32))
        else:
            raise NotImplementedError('non-PCA/VAE encodings not done yet')
        
        # Number of predictor variables (not just N as with other ESN architectures):
        # outputs of encoders + inputs + last reservoir's states are used as predictors
        self.num_predic_vars = np.sum(len(self.encoders) * np.array(self.dim_reduce))
        self.num_predic_vars += self.K + self.reservoirs[-1].N

    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        self.reservoirs.append(Reservoir(self.K, reservoir_sizes[0], echo_params[0],
                                         idx=0, debug=self.debug))
        for i, (size, echo_prm) in enumerate(zip(reservoir_sizes, echo_params)[1:]):
            self.reservoirs.append(Reservoir(
                input_size=self.dim_reduce, num_units=size, echo_param=echo_prm,
                idx=i+1, activation=activation, debug=self.debug
            ))

    def __forward_routing_rule__(self, u_n):
        x_n = np.zeros(0)

        for reservoir, encoder in zip(self.reservoirs, self.encoders):
            u_n = reservoir.forward(u_n)
            #u_n -= np.mean(u_n)
            if self.encoder_type == 'PCA':
                u_n = encoder.transform(u_n.reshape(1, -1)).squeeze()
            elif self.encoder_type == 'VAE':
                u_n = encoder.encode(Variable(th.FloatTensor(u_n)))[0].data.numpy()

            x_n = np.append(x_n, u_n)

        u_n = self.reservoirs[-1].forward(u_n)
        x_n = np.append(x_n, u_n)

        return x_n

    def train(self, X, y):
        """ 
        Needs different train() because reservoirs+encoders have to be warmed up+trained one at a time.
        
        WARNING: Training a DHESN requires 'init_echo_timesteps' of initial warm-up/burn-in 
                 timesteps for EACH reservoir, e.g. a DHESN with 5 reservoirs and 4 encoders 
                 requires FIVE-HUNDRED initial timesteps (for 'init_echo_timesteps' = 100).
        """
        assert X.shape[1] == self.K, "Training data has unexpected dimensionality (%s). K = %d." % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)
        assert self.encoder_type != 'PCA' or np.mean(X) < 1e-3, "Input data must be zero-mean to use PCA encoding."

        T = len(X) - self.init_echo_timesteps*self.num_reservoirs
        # S = np.zeros((T, self.N+self.K))
        # S = np.zeros((T, 5))
        S = np.zeros((T, (self.num_reservoirs-1)*self.dim_reduce+self.K+self.reservoirs[-1].N))
        # S: collection of extended system states (encoder outputs plus inputs)
        #     at each time-step t
        S[:, -self.K:] = X[self.init_echo_timesteps*self.num_reservoirs:]
        delim = np.array([0]+[self.dim_reduce]*(self.num_reservoirs-1)+[self.reservoirs[-1].N])
        for i in range(1, len(delim)):
            delim[i] += delim[i-1]
            
        burn_in = X[:self.init_echo_timesteps] # feed a unique input set to all reservoirs
        inputs = X[self.init_echo_timesteps:]
        # Now send data into each reservoir one at a time,
        #   and train each encoder one at a time
        for i in range(self.num_reservoirs):
            reservoir = self.reservoirs[i]
            # burn-in period (init echo timesteps) ===============================================
            for u_n in burn_in:
                _ = reservoir.forward(u_n)
            # ===========================================================================

            N_i = reservoir.N         # number of units in reservoir i
            S_i = np.zeros((T, N_i))  # reservoir i's states over T timesteps

            # Now collect the real state data for encoder to train on ===================
            for n, u_n in enumerate(inputs):
                S_i[n, :] = reservoir.forward(u_n)

            # All reservoirs except the last output into an autoencoder =================
            if i != self.num_reservoirs - 1:
                encoder = self.encoders[i]
                # Now train the encoder using the gathered state data
                if self.encoder_type == 'PCA':
                    S_i -= np.mean(S_i)
                    encoder.fit(S_i)
                    S_i = encoder.transform(S_i)
                elif self.encoder_type == 'VAE':
                    # encoder.train_full(Variable(th.FloatTensor(S_i)))
                    # S_i = encode.encode(S_i).data().numpy()
                    encoder.train_full(th.FloatTensor(S_i))
                    S_i = encoder.encode(Variable(th.FloatTensor(S_i)))[0].data.numpy()

            # first few are for the next burn-in
            burn_in = S_i[:self.init_echo_timesteps, :]
            # rest are the next inputs
            inputs = S_i[self.init_echo_timesteps:, :]

            if self.debug:
                print('inputs shape', np.shape(inputs))
                print('S_i shape', np.shape(S_i))
                print('S shape', np.shape(S))

            # Slot the state data into its corresponding spot in the full state matrix ==
            lb, ub = delim[i], delim[i+1]
            S[:, lb:ub] = S_i[(self.init_echo_timesteps*(self.num_reservoirs-i-1)):, :]
            
            if self.debug:
                print('Mean state magnitude of res. %d: %.4f' % (i, np.mean(np.abs(S_i))))

        D = y[self.init_echo_timesteps*self.num_reservoirs:]
        # Solve linear system
        T1 = np.dot(D.T, S)
        T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(self.num_predictor_variables))
        self.W_out = np.dot(T1, T2)
        
    @property
    def num_predictor_variables(self):
        return self.num_predic_vars


class LCESN(LayeredESN):
    """ Layered constrained ESN (name probably needs a change). """
    
    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        """
        Set up the reservoirs so that the first takes the input signal as input,
          and the rest take the previous reservoir's state as input.
        """
        self.reservoirs.append(Reservoir(self.K, reservoir_sizes[0], echo_params[0],
                                         idx=0, activation=activation, debug=self.debug))
        for i, (size, echo_prm) in enumerate(zip(reservoir_sizes, echo_params)[1:]):
            self.reservoirs.append(Reservoir(
                input_size=self.reservoirs[i-1].N, num_units=size, echo_param=echo_prm,
                idx=i+1, activation=activation, debug=self.debug
            ))

    def __forward_routing_rule__(self, u_n):
        x_n = np.zeros(0)
        for reservoir in self.reservoirs:
            u_n = reservoir.forward(u_n)
            x_n = np.append(x_n, u_n)

        return x_n


class EESN(LayeredESN):
    """ Ensemble (of reservoirs) echo state network. """

    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        """
        Set up the reservoirs so that they all take the input signal as input.
        """
        for i, (size, echo_prm) in enumerate(zip(reservoir_sizes, echo_params)):
            self.reservoirs.append(Reservoir(
                input_size=self.K, num_units=size, echo_param=echo_prm,
                idx=i, activation=activation, debug=self.debug
            ))

    def __forward_routing_rule__(self, u_n):
        x_n = np.zeros(0)
        for reservoir in self.reservoirs:
            output = reservoir.forward(u_n)
            x_n = np.append(x_n, output)

        return x_n
                    
