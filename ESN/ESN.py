import numpy as np
import numpy.linalg as la
import pickle as pkl
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
        self.sparsity = None
        self.W_res_init_strategy = None
        self.input_weights_scale = None
        self.W_in_init_strategy = None
        self.sparsity = None

        # helpful information to track
        #if self.debug:
        self.signals = [] # <- reservoir states over time during training
        self.num_to_store = 50
        self.ins_init = False; self.res_init = False

    def info(self):
        out = u'Reservoir(N=%d, K=%d, \u03B5=%.2f)\n' % (self.N, self.K, self.echo_param)
        out += 'W_res - spec_scale: %.2f, %s init\n' % (self.spectral_scale, self.W_res_init_strategy)
        out += 'W_in  -      scale: %.2f, %s init' % (self.input_weights_scale, self.W_in_init_strategy)

    def initialize_input_weights(self, strategy='binary', scale=1e-2, offset=0.5, sparsity=1.0):
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

        self.sparsity_input = sparsity
        sparsity_matrix = (np.random.rand(self.N, self.K) < self.sparsity_input).astype(float)

        self.W_in -= offset
        self.W_in *= sparsity_matrix
        self.W_in *= self.input_weights_scale
        self.ins_init = True

    def initialize_reservoir_weights(self, strategy='uniform', spectral_scale=1.0, offset=0.5, 
                                     sparsity=1.0):
        self.spectral_scale = spectral_scale
        self.W_res_init_strategy = strategy
        self.sparsity = sparsity
        if strategy == 'binary':
            self.W_res = (np.random.rand(self.N, self.N) > 0.5).astype(float)
        elif strategy == 'uniform':
            self.W_res = np.random.rand(self.N, self.N)
        elif strategy == 'gaussian':
            self.W_res = np.random.randn(self.N, self.N)
        else:
            raise ValueError('unknown res. weight init strategy %s' % strategy)

        # apply the sparsity
        self.sparsity = sparsity
        sparsity_matrix = (np.random.rand(self.N, self.N) < self.sparsity).astype(float)

        self.W_res -= offset
        self.W_res *= sparsity_matrix
        self.W_res /= max(abs(la.eig(self.W_res)[0]))
        self.W_res *= self.spectral_scale
        self.res_init = True

    def forward(self, u_n):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        u_n: K-dimensional input vector
        """
        u_n = u_n.squeeze()
        try:
            assert (self.K == 1 and u_n.shape == ()) or u_n.shape[0] == self.W_in.shape[1], \
                "u(n): %s.  W_res: %s (ID=%d)" % (u_n.shape, self.W_res.shape, self.idx)
        except:
            print(u_n.shape)
            print(self.W_in.shape)
            print(self.W_res.shape)
            raise
        assert self.ins_init, "Res. input weights not yet initialized (ID=%d)." % self.idx
        assert self.res_init, "Res. recurrent weights not yet initialized (ID=%d)." % self.idx

        in_to_res = np.dot(self.W_in, u_n).squeeze()
        res_to_res = np.dot(self.state.reshape(1, -1), self.W_res)

        # Equation (1) in "Formalism and Theory" of Scholarpedia page
        self.state = (1. - self.echo_param) * self.state + self.echo_param * self.activation(in_to_res + res_to_res)
        #if self.debug:
        self.signals.append(self.state[:self.num_to_store].tolist())

        return self.state.squeeze()


class ESN(object):

    def __init__(self, input_size, output_size, reservoir_size, echo_param=0.6,
                 output_activation=None, init_echo_timesteps=100, regulariser=1e-8,
                 activation=np.tanh, debug=False):
        # IMPLEMENTATION STUFF ===================================================
        if input_size != output_size:
            raise NotImplementedError('num input dims must equal num output dims.')
        if output_activation is not None:
            raise NotImplementedError('non-identity output activations not implemented.')
        # ========================================================================
        self.K = input_size
        self.N = reservoir_size
        self.L = output_size
        self.reservoir = Reservoir(input_size=self.K, num_units=self.N, 
                                    echo_param=echo_param, activation=activation, debug=debug)
        if output_activation is None:
            def iden(x): return x
            output_activation = iden    # <- identity
        self.init_echo_timesteps = init_echo_timesteps
        self.regulariser = regulariser
        self.output_activation = output_activation
        self.debug = debug

        self.W_out = np.ones((self.L, self.K+self.N))   # output weights

    def info(self):
        r_size = self.reservoir.N
        e_prm = self.reservoir.echo_param
        i_scl = self.reservoir.input_weights_scale
        s_scl = self.reservoir.spectral_scale
        sp = self.reservoir.sparsity
        reg = self.regulariser
        out = 'r_size:%d\ne_prm:%f\ni_scl:%f\ns_scl:%f\nsp:%f\nreg:%f' % \
                (r_size, e_prm, i_scl, s_scl, sp, reg)
        return out

    def initialize_input_weights(self, strategy='binary', scale=1e-2, sparsity=1.0):
        self.reservoir.initialize_input_weights(strategy, scale, sparsity=1.0)

    def initialize_reservoir_weights(self, strategy='uniform', spectral_scale=1.0, offset=0.5, 
                                     sparsity=1.0):
        self.reservoir.initialize_reservoir_weights(strategy, spectral_scale, offset, sparsity)

    def forward(self, u_n, add_bias=True):
        u_n = u_n.squeeze()

        assert (self.K == 1 and u_n.shape == ()) or len(u_n) == self.K, "unexpected input dimensionality (check bias)"

        x_n = self.reservoir.forward(u_n)  # reservoir states at time n
        z_n = np.append(x_n, u_n)          # extended system states at time n
        if add_bias:
            z_n = np.hstack((z_n, 1)) 

        # by default, output activation is identity
        # output = self.output_activation(np.dot(z_n, self.W_out.T))
        output = self.output_activation(np.dot(self.W_out, z_n))

        return output.squeeze()

    def train(self, X, y, add_bias=True):
        assert X.shape[1] == self.K, "training data has unexpected dimensionality (%s); K = %d" % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)

        # First, run a few inputs into the reservoir to get it echoing
        initial_data = X[:self.init_echo_timesteps]
        for u_n in initial_data:
            _ = self.reservoir.forward(u_n)

        if self.debug: print('-'*10+'Initial echo timesteps done. '+'-'*10)

        # Now train the output weights
        X_train = X[self.init_echo_timesteps:]
        D = y[self.init_echo_timesteps:]                  # <- teacher output collection matrix
        S = np.zeros((X_train.shape[0], self.N + self.K)) # <- state collection matrix
        for n, u_n in enumerate(X_train):
            x_n = self.reservoir.forward(u_n)
            z_n = np.append(x_n, u_n)
            S[n, :] = z_n
        if self.debug: print('-'*10+'Extended system states collected.'+'-'*10)

        if add_bias:
            S = np.hstack([S, np.ones((S.shape[0], 1))])
        # Solve (W_out)(S.T) = (D) by least squares
        T1 = np.dot(D.T, S)                                                       # L     x (N+K)
        reg = self.regulariser * np.eye(self.K + self.N+1)
        reg[-1, -1] = 0
        T2 = la.inv(np.dot(S.T, S) + reg)  # (N+K) x (N+K)
        self.W_out = np.dot(T1, T2)                                               # L     x (N+K)
        
    def reset_reservoir_states(self):
        self.reservoir.state = np.zeros(self.N)

    def getInputSize(self): return self.K

    def getOutputSize(self): return self.L

class LayeredESN(object):
    """
    (ABSTRACT CLASS)
    Layered echo state network (LESN).

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

    def initialize_input_weights(self, strategies='binary', scales=2e-2, offsets=0.5, sparsity=1.0):
        if type(strategies) not in [list, np.ndarray]:
            strategies = [strategies]*self.num_reservoirs
        if type(scales) not in [list, np.ndarray]:
            scales = [scales]*self.num_reservoirs
        if type(offsets) not in [list, np.ndarray]:
            offsets = [offsets]*self.num_reservoirs

        for i, (strat, scale) in enumerate(zip(strategies, scales)):
            self.reservoirs[i].initialize_input_weights(strategy=strat, scale=scale, sparsity=sparsity)

    def initialize_reservoir_weights(self, strategies='uniform', spectral_scales=1.0, offsets=0.5, sparsity=1.0, sparsities=None):
        if type(strategies) not in [list, np.ndarray]:
            strategies = [strategies]*self.num_reservoirs
        if type(spectral_scales) not in [list, np.ndarray]:
            spectral_scales = [spectral_scales]*self.num_reservoirs
        if type(offsets) not in [list, np.ndarray]:
            offsets = [offsets]*self.num_reservoirs
        if sparsities is not None:
            assert len(sparsities) == self.num_reservoirs
        else:
            sparsities = [sparsity]*self.num_reservoirs

        for i, (strat, scale, offset, sp) in enumerate(
            zip(strategies, spectral_scales, offsets, sparsities)
        ):
            self.reservoirs[i].initialize_reservoir_weights(strat, scale, offset, sparsity=sp)

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

    def forward(self, u_n, calculate_output=True, add_bias=True):
        """
        Forward-propagate signal through network.
        If calculate_output = True: returns output signal, y_n.
                              else: returns updated system states, x_n.
        """
        u_n = u_n.squeeze()

        assert (self.K == 1 and u_n.shape == ()) or  len(u_n) == self.K

        x_n = self.__forward_routing_rule__(u_n)

        if add_bias:
            u_n = np.hstack((u_n, 1))

        if calculate_output:
            z_n = np.append(x_n, u_n)
            output = self.output_activation(np.dot(self.W_out, z_n))
            return output.squeeze()
        else:
            return x_n

    def train(self, X, y, add_bias=True):

        assert X.shape[1] == self.K, "Training data has unexpected dimensionality (%s). K = %d." % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)

        # First, run a few inputs into the reservoir to get it echoing
        initial_data = X[:self.init_echo_timesteps]
        for u_n in initial_data:
            _ = self.forward(u_n, calculate_output=False, add_bias=False)

        # Now train the output weights
        X_train = X[self.init_echo_timesteps:]
        D = y[self.init_echo_timesteps:]
        S = np.zeros((X_train.shape[0], self.N+self.K))
        for n, u_n in enumerate(X_train):
            x_n = self.forward(u_n, calculate_output=False, add_bias=False)
            z_n = np.append(x_n, u_n)
            S[n, :] = z_n

        if add_bias:
            S = np.hstack([S, np.ones((S.shape[0], 1))])

        # Solve linear system
        T1 = np.dot(D.T, S)
        T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(self.K + self.N+1))
        self.W_out = np.dot(T1, T2)
        
    def reset_reservoir_states(self):
        for reservoir in self.reservoirs:
            reservoir.state *= 0.

    def getInputSize(self): return self.K

    def getOutputSize(self): return self.L

    def info(self):
        inp_scales = [r.input_weights_scale for r in self.reservoirs]
        spec_scales = [r.spectral_scale for r in self.reservoirs]
        echo_prms = [r.echo_param for r in self.reservoirs]
        sps = [r.sparsity for r in self.reservoirs]
        out = """
        num_res: %d\nres_sizes:%s\necho_params:%s\ninput_scales:%s\nspectral_scales:%s
        """ % (self.num_reservoirs, self.reservoir_sizes, echo_prms, inp_scales, spec_scales)
        out += 'sparsities:%s\nregulariser: %f' % (sps, self.regulariser)
        return out

class DHESN(LayeredESN):

    def __init__(self, *args, **kwargs):
        assert 'dims_reduce' in kwargs.keys() or kwargs['dims_reduce'] is list, "MUST UNCLUDE DIMS AS LIST."
        self.dims_reduce = kwargs['dims_reduce']
        del kwargs['dims_reduce']

        if 'train_epochs' not in kwargs.keys():
            self.train_epochs = 2 # should make this specific to only VAEs but being quick for now
        else:
            self.train_epochs = kwargs['train_epochs']
            del kwargs['train_epochs']

        if 'train_batches' not in kwargs.keys():
            self.train_batches = 64 # should make this specific to only VAEs but being quick for now
        else:
            self.train_batches = kwargs['train_batches']
            del kwargs['train_batches']

        if 'encoder_type' not in kwargs.keys():
            self.encoder_type = 'PCA'
        else:
            self.encoder_type = kwargs['encoder_type']
            del kwargs['encoder_type']

        if 'encode_norm' not in kwargs.keys():
            self.encode_norm = False # similar to batch norm (without trained std/mean) - we normalise AFTER the encoding
        else:
            self.encode_norm = kwargs['encode_norm']
            del kwargs['encode_norm']
        
        super(DHESN, self).__init__(*args, **kwargs)
        
        # print(self.dims_reduce)
        self.data_mean = None
        # normalisation data for reservoir outputs
        self.reservoir_means = [
            np.zeros(N_i) for N_i in self.reservoir_sizes
        ]
        self.reservoir_stds = [
            np.zeros(N_i) for N_i in self.reservoir_sizes
        ]
        # normalisation data for encoder outputs
        self.encoder_means = [
            np.zeros(N_i) for N_i in self.dims_reduce
        ]
        self.encoder_stds = [
            np.zeros(N_i) for N_i in self.dims_reduce
        ]

        self.encoders = []

        if self.encoder_type == 'PCA':
            for j in range(1, self.num_reservoirs):
                # self.encoders.append(PCA(n_components=self.reservoirs[j-1].N))
                self.encoders.append(PCA(n_components=self.dims_reduce[j-1]))
        elif self.encoder_type == 'VAE':
            for j in range(1, self.num_reservoirs):
                self.encoders.append(VAE(input_size=self.reservoir_sizes[j-1], latent_variable_size=self.dims_reduce[j-1],
                                            epochs=self.train_epochs, batch_size=self.train_batches))
                                            # epochs=self.train_epochs*j, batch_size=self.train_batches))
        else:
            raise NotImplementedError('non-PCA/VAE encodings not done yet')

        # signals of the encoders
        self.encoder_signals = [[] for _ in range(self.num_reservoirs-1)]

    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        self.reservoirs.append(Reservoir(self.K, reservoir_sizes[0], echo_params[0],
                                         idx=0, debug=self.debug))
        for i, (size, echo_prm) in enumerate(zip(reservoir_sizes, echo_params)[1:]):
            # self.reservoirs.append(Reservoir(
            #     input_size=self.reservoirs[i].N, num_units=size, echo_param=echo_prm,
            #     idx=i+1, activation=activation, debug=self.debug
            # ))
            self.reservoirs.append(Reservoir(
                input_size=self.dims_reduce[i], num_units=size, echo_param=echo_prm,
                idx=i+1, activation=activation, debug=self.debug
            ))

    def __forward_routing_rule__(self, u_n):
        x_n = np.zeros(0)

        # u_n = (u_n.reshape(-1, self.K) - self.data_mean).squeeze()

        for i, (reservoir, encoder) in enumerate(zip(self.reservoirs, self.encoders)):
            u_n = np.array(reservoir.forward(u_n))
            # u_n -= self.reservoir_means[i]

            if self.encoder_type == 'PCA':
                # normalising prior to PCA could be good of bad (https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca)
                #u_n -= self.reservoir_means[i]
                #u_n /= self.reservoir_stds[i]
                u_n = encoder.transform(u_n.reshape(1, -1)).squeeze()
            elif self.encoder_type == 'VAE':
                # we must normalise the input prior to VAE input
                u_n -= self.reservoir_means[i]
                u_n /= self.reservoir_stds[i]
                u_n = encoder.encode(Variable(th.FloatTensor(u_n)))[0].data.numpy()

            # normalise the outputs of the encoders
            if self.encode_norm:
                #print(np.shape(u_n))
                #print(np.shape(self.encoder_means[i]))
                #print(np.shape(self.encoder_stds[i]))
                u_n = np.array((u_n - self.encoder_means[i])/self.encoder_stds[i])
                #u_n -= self.encoder_means[i]
                #u_n /= self.encoder_stds[i]

            # store the encoded signals of each encoder
            self.encoder_signals[i].append(u_n.tolist())

            x_n = np.append(x_n, u_n)

        u_n = self.reservoirs[-1].forward(u_n)
        x_n = np.append(x_n, u_n)

        return x_n

    def train(self, X, y, debug_info=False, add_bias=True):
        """ (needs different train() because reservoirs+encoders have to be warmed up+trained one at a time."""
        assert X.shape[1] == self.K, "Training data has unexpected dimensionality (%s). K = %d." % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)
        #assert self.encoder_type != 'PCA' or np.mean(X) < 1e-3, "Input data must be zero-mean to use PCA encoding."
        # print("X mean: {}, y mean: {}".format(np.mean(X, axis=0), np.mean(y, axis=0)))
        self.data_mean = np.mean(X, axis=0)[0]
        # plt.plot(range(np.shape(X)[0]), X, label="before")
        # print("mean: {}".format(self.data_mean))
        # print(np.shape(X))
        # print(np.shape(y))
        # print(np.shape(self.data_mean))
        # X -= self.data_mean
        # y -= self.data_mean
        # print("X mean: {}, y mean: {}".format(np.mean(X, axis=0), np.mean(y, axis=0)))
        # plt.plot(range(np.shape(X)[0]), X, label="after")
        # plt.show()

        T = len(X) - self.init_echo_timesteps*self.num_reservoirs
        # S = np.zeros((T, self.N+self.K))
        # S = np.zeros((T, 5))
        S = np.zeros((T, np.sum(self.dims_reduce)+self.K+self.reservoirs[-1].N))
        # S: collection of extended system states (encoder outputs plus inputs)
        #     at each time-step t
        S[:, -self.K:] = X[self.init_echo_timesteps*self.num_reservoirs:]
        # delim = np.array([0]+[r.N for r in self.reservoirs])
        delim = np.array([0]+self.dims_reduce+[self.reservoirs[-1].N])
        for i in range(1, len(delim)):
            delim[i] += delim[i-1]
            
        # inputs = X[:self.init_echo_timesteps, :]
        # inputs_next = X[self.init_echo_timesteps:(self.init_echo_timesteps*2), :]
        burn_in = X[:self.init_echo_timesteps] # feed a unique input set to all reservoirs
        inputs = X[self.init_echo_timesteps:]
        # Now send data into each reservoir one at a time,
        #   and train each encoder one at a time
        for i in range(self.num_reservoirs):
            reservoir = self.reservoirs[i]
            # burn-in period (init echo timesteps) ===============================================
            for u_n in burn_in:
                _ = reservoir.forward(u_n)
            # ==================

            N_i = reservoir.N
            S_i = np.zeros((np.shape(inputs)[0], N_i))  # reservoir i's states over T timesteps

            # Now collect the real state data for encoders to train on
            for n, u_n in enumerate(inputs):
                S_i[n, :] = reservoir.forward(u_n)

            # All reservoirs except the last output into an autoencoder
            if i != self.num_reservoirs - 1:
                encoder = self.encoders[i]
                res_mean = np.mean(S_i, axis=0)
                res_std = np.std(S_i, axis=0) + 1e-8
                # print("MEAN: {}".format(res_mean))
                # print("STD: {}".format(res_std))
                # print(res_mean)
                self.reservoir_means[i] = res_mean
                self.reservoir_stds[i] = res_std
                # S_i -= res_mean
                # Now train the encoder using the gathered state data
                if self.encoder_type == 'PCA':
                    encoder.fit(S_i)  # sklearn PCA automatically zero-means the data
                    S_i = np.array(encoder.transform(S_i))
                elif self.encoder_type == 'VAE':
                    # print(S_i[:3])
                    S_i -= res_mean
                    # print(S_i[:3])
                    # print("="*10)
                    S_i /= res_std
                    # encoder.train_full(Variable(th.FloatTensor(S_i)))
                    # S_i = encode.encode(S_i).data().numpy()
                    S_i_train = np.array(S_i[:-1000, :])
                    S_i_test = np.array(S_i[-1000:, :])
                    encoder.train_full(th.FloatTensor(S_i_train), th.FloatTensor(S_i_test))
                    S_i = np.array(encoder.encode(Variable(th.FloatTensor(S_i)))[0].data.numpy())
                    # S_i = encoder.encode(Variable(th.FloatTensor(S_i))).data.numpy()
                # S_i += res_mean[:100]

                # compute the mean output of the encoders
                enc_mean = np.mean(S_i, axis=0)
                enc_std = np.std(S_i, axis=0)+1e-8
                self.encoder_means[i] = np.array(enc_mean) # this would be ~0 anyway because we normalise prior to encoding (but still...)
                self.encoder_stds[i] = np.array(enc_std)

            # first few are for the next burn-in
            burn_in = S_i[:self.init_echo_timesteps, :]
            # rest are the next inputs
            inputs = S_i[self.init_echo_timesteps:, :]

            # print(np.shape(inputs))
            # print(np.shape(S_i))
            # print(np.shape(S))
            lb, ub = delim[i], delim[i+1]
            S[:, lb:ub] = S_i[(self.init_echo_timesteps*(self.num_reservoirs-i-1)):, :]

            # inputs = S_i
            
            if debug_info:
                print('res %d mean state magnitude: %.4f' % (i, np.mean(np.abs(S_i))))

        if add_bias:
            S = np.hstack([S, np.ones((S.shape[0], 1))])

        D = y[self.init_echo_timesteps*self.num_reservoirs:]
        # Solve linear system
        T1 = np.dot(D.T, S)
        # T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(self.K + self.N))
        T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(np.sum(self.dims_reduce)+self.K+self.reservoirs[-1].N+1))
        self.W_out = np.dot(T1, T2)

    @property
    def input_size(self):
        return self.K
    
    @property
    def output_size(self):
        return self.L


class LCESN(LayeredESN):
    
    def __reservoir_input_size_rule__(self, reservoir_sizes, echo_params, activation):
        """
        Set up the reservoirs so that the first takes the input signal as input,
          and the rest take the previous reservoir's state as input.
        """
        self.reservoirs.append(Reservoir(self.K, reservoir_sizes[0], echo_params[0],
                                         idx=0, debug=self.debug))
        for i, (size, echo_prm) in enumerate(zip(reservoir_sizes, echo_params)[1:]):
            self.reservoirs.append(Reservoir(
                input_size=self.reservoirs[i].N, num_units=size, echo_param=echo_prm,
                idx=i+1, activation=activation, debug=self.debug
            ))

    def __forward_routing_rule__(self, u_n):
        x_n = np.zeros(0)
        for reservoir in self.reservoirs:
            u_n = reservoir.forward(u_n)
            x_n = np.append(x_n, u_n)

        return x_n


class EESN(LayeredESN):

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

class EESN_ENCODED(LayeredESN):

    def __init__(self, *args, **kwargs):
        assert 'dims_reduce' in kwargs.keys() or kwargs['dims_reduce'] is list, "MUST UNCLUDE DIMS AS LIST."
        self.dims_reduce = kwargs['dims_reduce']
        del kwargs['dims_reduce']

        if 'train_epochs' not in kwargs.keys():
            self.train_epochs = 2 # should make this specific to only VAEs but being quick for now
        else:
            self.train_epochs = kwargs['train_epochs']
            del kwargs['train_epochs']

        super(EESN_ENCODED, self).__init__(*args, **kwargs)
        
        self.data_mean = None
        # normalisation data for reservoir outputs
        self.reservoir_means = [
            np.zeros(N_i) for N_i in self.reservoir_sizes
        ]
        self.reservoir_stds = [
            np.zeros(N_i) for N_i in self.reservoir_sizes
        ]
        # normalisation data for encoder outputs
        self.encoder_means = [
            np.zeros(N_i) for N_i in self.dims_reduce
        ]
        self.encoder_stds = [
            np.zeros(N_i) for N_i in self.dims_reduce
        ]

        self.encoders = []

        for j in range(self.num_reservoirs):
            self.encoders.append(VAE(input_size=self.reservoir_sizes[j], latent_variable_size=self.dims_reduce[j],
                                        epochs=self.train_epochs, batch_size=64))

        # signals of the encoders
        self.encoder_signals = [[] for _ in range(self.num_reservoirs)]


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

        for i, (reservoir, encoder) in enumerate(zip(self.reservoirs, self.encoders)):
            output = np.array(reservoir.forward(u_n))
            output -= self.reservoir_means[i]
            output /= self.reservoir_stds[i]
            output = np.array(encoder.encode(Variable(th.FloatTensor(output)))[0].data.numpy())

            # store the encoded signals of each encoder
            self.encoder_signals[i].append(output.tolist())

            x_n = np.append(x_n, output)

        return x_n

    def train(self, X, y, debug_info=False, add_bias=True):
        """ (needs different train() because reservoirs+encoders have to be warmed up+trained one at a time."""
        assert X.shape[1] == self.K, "Training data has unexpected dimensionality (%s). K = %d." % (X.shape, self.K)
        X = X.reshape(-1, self.K)
        y = y.reshape(-1, self.L)
        self.data_mean = np.mean(X, axis=0)[0]

        T = len(X) - self.init_echo_timesteps
        S = np.zeros((T, np.sum(self.dims_reduce)+self.K))
        S[:, -self.K:] = X[self.init_echo_timesteps:]
        delim = np.array([0]+self.dims_reduce)
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
            # ==================

            N_i = reservoir.N
            S_i = np.zeros((np.shape(inputs)[0], N_i))  # reservoir i's states over T timesteps

            # Now collect the real state data for encoders to train on
            for n, u_n in enumerate(inputs):
                S_i[n, :] = reservoir.forward(u_n)

            # All reservoirs except the last output into an autoencoder
            encoder = self.encoders[i]
            res_mean = np.mean(S_i, axis=0)
            res_std = np.std(S_i, axis=0) + 1e-8

            self.reservoir_means[i] = res_mean
            self.reservoir_stds[i] = res_std

            S_i -= res_mean
            S_i /= res_std
            S_i_train = np.array(S_i[:-1000, :])
            S_i_test = np.array(S_i[-1000:, :])
            encoder.train_full(th.FloatTensor(S_i_train), th.FloatTensor(S_i_test))
            #encoder.train_full(th.FloatTensor(S_i))
            S_i = np.array(encoder.encode(Variable(th.FloatTensor(S_i)))[0].data.numpy())

            enc_mean = np.mean(S_i, axis=0)
            enc_std = np.std(S_i, axis=0)+1e-8
            self.encoder_means[i] = np.array(enc_mean) # this would be ~0 anyway because we normalise prior to encoding (but still...)
            self.encoder_stds[i] = np.array(enc_std)

            lb, ub = delim[i], delim[i+1]
            S[:, lb:ub] = np.array(S_i)

            # inputs = S_i
            
            if debug_info:
                print('res %d mean state magnitude: %.4f' % (i, np.mean(np.abs(S_i))))

        if add_bias:
            S = np.hstack([S, np.ones((S.shape[0], 1))])

        D = y[self.init_echo_timesteps:]
        # Solve linear system
        T1 = np.dot(D.T, S)
        # T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(self.K + self.N))
        T2 = la.inv(np.dot(S.T, S) + self.regulariser * np.eye(np.sum(self.dims_reduce)+self.K+1))
        self.W_out = np.dot(T1, T2)


class ESN2(object):
    """
    Echo state network  -------------OLD ONE-----------------.
    
    N = reservoir_size; K = input_size; L = output_size
    Dimensions, notation guide:
         W_in: (N x K)        (inputs-to-reservoir weight matrix)
            W: (N x N)        (reservoir-to-reservoir weight matrix)
        W_out: (L x (K+N))    (reservoir-to-output weight matrix)

         u(n): K-dimensional input signal at time n.
         x(n): N-dimensional reservoir states at time n.
         y(n): L-dimensional output signal at time n.
         d(n): L-dimensional TRUE output signal at time n.
         z(n): (N+K)-dimensional extended system states at time n, [x(n); u(n)].

            f: Activation function for the reservoir units.
            g: Activation function for the output layer (possibly identity).
    """

    def __init__(self, input_size, output_size, reservoir_size=100, echo_param=0.6, 
                 spectral_scale=1.0, init_echo_timesteps=100,
                 regulariser=1e-8, input_weights_scale=(1/100.),
                 debug_mode=False):

        # np.random.seed(42)
        # ARCHITECTURE PARAMS
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.activation_function = np.tanh
        self.input_weights_scale = input_weights_scale

        # RESERVOIR PARAMS
        self.spectral_scale = spectral_scale
        self.reservoir_state = np.zeros((1, self.reservoir_size))
        self.echo_param = echo_param
        self.init_echo_timesteps = init_echo_timesteps # number of inititial runs before training
        self.regulariser = regulariser

        # WEIGHTS
        #self.W_in = (np.random.randn(input_size, reservoir_size) - 0.5)*(1/1000.)
        self.W_in = ((np.random.rand(input_size, reservoir_size) > 0.5).astype(int) - 0.5) *self.input_weights_scale 
        #self.W_in = (np.random.rand(input_size, reservoir_size) - 0.5) *self.input_weights_scale 

        # Reservoir-to-reservoir weights (N x N)
        self.W_reservoir = []
        # self.__reservoir_norm_spectral_radius_norm_weights__()
        self.__reservoir_norm_spectral_radius_uniform_weights__()

        #self.W_reservoir = np.random.rand(self.reservoir_size, self.reservoir_size)-0.5

        # Reservoir-to-output weights (L x (K+N))
        self.W_out = []

        self.debug = debug_mode

        if self.debug: print("W_in[:10]: {}".format(self.W_in[:10]))
        if self.debug: print("W_res: {}".format(self.W_reservoir))

        # SOME EXTA STORE DATA
        self.training_signals = [] # reservoir state over time during training

    def copy(self):
        return ESN2(self.input_size, self.output_size, self.reservoir_size, self.echo_param,
                    self.spectral_scale, self.init_echo_timesteps,
                    self.regulariser, self.input_weights_scale, self.debug)

    def reset_reservoir(self):
        """ Reset reservoir states to zeros (does not reset W_out weights). """
        self.reservoir_state = np.zeros((1, self.reservoir_size))

    def __reservoir_norm_spectral_radius_norm_weights__(self):
        """ Initialize reservoir weights using standard normal Gaussian. """
        return self.__reservoir_norm_spectral_radius__(np.random.randn)

    def __reservoir_norm_spectral_radius_uniform_weights__(self):
        """ Initialize reservoir weights using uniform [0, 1]. """
        return self.__reservoir_norm_spectral_radius__(np.random.rand)

    def __reservoir_norm_spectral_radius_binary_weights__(self):
        """ Initialize reservoir weights u.a.r. from {0, 1}. """
        def binary_distr(d0, d1):
            return (np.random.rand(d0, d1) + 0.5).astype(int)
        return self.__reservoir_norm_spectral_radius__(binary_distr)

    def __reservoir_norm_spectral_radius__(self, weight_distribution_function, offset=0.5):
        """ 
        Initializes the reservoir weights according to some initialization strategy 
            (e.g. uniform in [0, 1], standard normal).
        Then, sets its spectral radius = desired value.
        """
        # self.W_reservoir = np.random.rand(reservoir_size, reservoir_size)
        self.W_reservoir = weight_distribution_function(self.reservoir_size, self.reservoir_size) - offset
        # make the spectral radius < 1 by dividing by the absolute value of the largest eigenvalue.
        self.W_reservoir /= max(abs(np.linalg.eig(self.W_reservoir)[0]))
        self.W_reservoir *= self.spectral_scale

    def __forward_to_res__(self, x_in):
        """ x_in = u(n). Puts input signal u(n) into reservoir, returns reservoir states x(n). """

        assert np.shape(x_in)[1] == np.shape(self.W_in)[0], "input of {} does not match input weights of {}".format(np.shape(x_in)[1], np.shape(self.W_in)[0])

        # in_to_res = W_in u(n+1)
        in_to_res = np.dot(x_in, self.W_in)
        # res_to_res = W x(n)
        res_to_res = np.dot(self.reservoir_state, self.W_reservoir)

        assert np.shape(in_to_res) == np.shape(res_to_res), "in-to-res input is {} whereas res-to-res input is {}".format(np.shape(in_to_res), np.shape(res_to_res))

        # E = echo parameter; f = activation function
        # x(n+1) = (1 - E) x(n) + E f(W x(n) + W_in u(n+1))
        self.reservoir_state = (1.0 - self.echo_param)*self.reservoir_state + self.echo_param*self.activation_function(in_to_res + res_to_res)
        
        #res_to_out = np.dot(self.reservoir_state, self.W_out)
        return self.reservoir_state.squeeze()

    def forward_to_out(self, x_in):
        """
        x_in = u(n).
        Puts input signal u(n) into reservoir; gets updated reservoir states x(n).
        Gets z(n) = [x(n); u(n)]. Returns y(n) = z(n) W_out.T
        """
        assert len(self.W_out) > 0, "ESN has not been trained yet!"
        assert len(np.shape(x_in)) == 1, "input should have only 1 dimension. Dimension is: {}".format(np.shape(x_in))

        # print(np.shape(x_in))
        res_out = np.array(self.__forward_to_res__(np.array([x_in])))
        x_n = res_out
        res_out = np.hstack((res_out, x_in)) # augment the data with the reservoir data
        # print(np.shape(res_out))
        assert np.shape(res_out)[0] == np.shape(self.W_out)[0], "res output is {}, whereas expected weights are {}".format(np.shape(res_out), np.shape(self.W_out))

        # z(n): (N+K); W_out.T: ((N+K)xL); y(n) = z(n) W_out.T
        res_to_out = np.dot(res_out, self.W_out)

        return res_to_out

    def train(self, data_X, data_y):

        # check that the data dimensions are the same as the input
        assert np.shape(data_X)[1] == self.input_size, "input data is {}; expected input size is {}".format(np.shape(data_X)[1], self.input_size)
        assert len(np.shape(data_X[0])) == 1, "input should have only 1 dimension"

        # first we run the ESN for a few inputs so that the reservoir starts echoing
        data_init = data_X[:self.init_echo_timesteps]
        data_train_X = data_X[self.init_echo_timesteps:]
        data_train_y = data_y[self.init_echo_timesteps:]
        for d in data_init:
            # print(d)
            _ = self.__forward_to_res__(np.array([d]))

        if self.debug: print("-"*10+"INITIAL ECHO TIMESTEPS DONE."+"-"*10)

        # now train the reservoir data after we have set up the echo state
        y_out = np.zeros((np.shape(data_train_X)[0], self.reservoir_size+self.input_size))
        for idx,d in enumerate(data_train_X):
            # print(np.shape(np.array([d])))
            y = self.__forward_to_res__(np.array([d]))
            y = np.hstack((y, d)) # augment the data with the reservoir data
            # print(np.shape(y))
            y_out[idx, :] = y
        if self.debug: print("-"*10+"DATA PUT THROUGH RESERVOIR DONE."+"-"*10)

        # do linear regression between the inputs and the output
        X_train = y_out
        y_target = data_train_y

        if self.debug: print("y: {}".format((y_target)))
        if self.debug: print("x: {}".format((X_train)))

        # plot some reservoir activations:
        if self.debug:
            num_signals = 10
            length_of_signal = 1000
            plt.plot(X_train[:length_of_signal, :num_signals])
            plt.title("Reservoir Signals for SPEC: {}, ECHO: {}".format(self.spectral_scale,
                        self.echo_param))
            plt.show()
        # store training signals for later analysis
        self.training_signals = X_train

        # X_reg = np.vstack((X_train, np.eye(self.reservoir_size+self.input_size)*self.regulariser))
        # y_reg = np.vstack((y_target, np.zeros((self.reservoir_size+self.input_size, 1))))

        # lsq_result = np.linalg.lstsq(X_reg, y_reg)
        T1 = np.dot(X_train.T, X_train) + self.regulariser*np.eye(self.input_size+self.reservoir_size)
        T2 = la.inv(np.dot(X_train.T, X_train) + self.regulariser*np.eye(self.input_size + self.reservoir_size))
        lsq_result = np.dot(np.dot(y_target.T, X_train), np.linalg.inv(np.dot(X_train.T,X_train) + \
                        self.regulariser*np.eye(self.input_size+self.reservoir_size)))
        self.W_out = lsq_result[0]
        if self.debug: print(self.W_out)

        if self.debug: print("W_out: {}".format(self.W_out))

        if self.debug: print("-"*10+"LINEAR REGRESSION ON OUTPUT DONE."+"-"*10)
        if self.debug: print("ESN trained!")

    def predict(self, data, reset_res=False):
        # We do not need to 'initialise' the ESN because the training phase already did this
        if reset_res:
            self.reset_reservoir()
            data_offset = self.init_echo_timesteps
        else:
            data_offset = 0

        y_out = np.zeros((np.shape(data)[0]-data_offset, 1))

        for idx,d in enumerate(data):
            if reset_res and idx < self.init_echo_timesteps:
                _ = self.forward_to_out(d)
            else:
                y = self.forward_to_out(d)
                y_out[idx-data_offset, :] = y

        return y_out
        #return data[:,0][:, None] - 0.01

    def generate(self, data, MEAN_OF_DATA, sample_step=None, plot=True, show_error=True):
        """ Pass the trained model. """
        # reset the reservoir
        self.reset_reservoir()
        #print(data)

        input_size = self.input_size-1 # -1 because of bias

        generated_data = []
        for i in range(0, len(data)-input_size):
            # run after the reservoir has "warmed-up"
            if i >= self.init_echo_timesteps:
                inputs = np.hstack((inputs, output[0]))
                inputs = inputs[1:]
                d_bias = np.hstack(([inputs], np.ones((1,1))))

                output = self.predict(d_bias)
                generated_data.append(output[0][0])
            # "warm-up" the reservoir
            else:
                inputs = data[i:(i+input_size), 0]
                d_bias = np.hstack(([inputs], np.ones((1,1))))
                output = self.predict(d_bias)

        if self.debug: print(np.shape(data[(self.init_echo_timesteps+input_size):]))
        if self.debug: print(np.shape(np.array(generated_data)[:, None]))
        if self.debug: print(np.hstack((data[(self.init_echo_timesteps+input_size):], 
                            np.array(generated_data)[:, None])))
        #error = np.mean((np.array(generated_data)[:, None] - data[(self.init_echo_timesteps+input_size):])**2)
        error = self.nmse(data[(self.init_echo_timesteps+input_size):], np.array(generated_data)[:, None], MEAN_OF_DATA)
        #error_mean = np.mean((np.array(generated_data)[:, None] - data[(1+input_size):])**2)
        #error_var_2 = np.sum((np.mean(data[(1+input_size):]) - data[(1+input_size):])**2)
        #error = (1.0 - error_mean/error_var_2)
        #error = np.mean((np.array(generated_data)[:, None] - data[(1+input_size):])**2)

        if show_error: print('NMSE generating test: %.7f' % error)

        if plot:
            xs = range(np.shape(data[self.init_echo_timesteps:])[0] - input_size)
            f, ax = plt.subplots()
            # print(np.shape(xs))
            # print(np.shape(data[(input_size+self.init_echo_timesteps):, 0]))
            #ax.plot(xs, data[(input_size+self.init_echo_timesteps):, 0], label='True data')
            ax.plot(range(len(generated_data)), data[(self.init_echo_timesteps+input_size):, 0], label='True data', c='red')
            ax.scatter(range(len(generated_data)), data[(self.init_echo_timesteps+input_size):, 0], s=4.5, c='black', alpha=0.5) 
            ax.plot(range(len(generated_data)), generated_data, label='Generated data', c='blue')
            ax.scatter(range(len(generated_data)), generated_data, s=4.5, c='black', alpha=0.5)
            # if sample_step is not None:
            #     smp_xs = np.arange(0, len(xs), sample_step)
            #     smp_ys = [data[x+input_size] for x in smp_xs]
            #     ax.scatter(smp_xs, smp_ys, label='sampling markers')
            # if show_error:
            #     ax.plot(xs, error, label='error')
            #     ax.plot(xs, [0]*len(xs), linestyle='--')
            plt.legend()
            plt.show()

        return error, generated_data


    def mean_l2_error(self, y_out, y_pred):
        if self.debug: print(np.hstack((y_out, y_pred)))
        return np.mean((np.array(y_out) - np.array(y_pred))**2)

    def nmse(self, y_out, y_pred, MEAN_OF_DATA):
        # y_out_mean = np.mean(y_out)
        return np.sqrt(np.sum((y_out - y_pred)**2)/np.sum((y_out - MEAN_OF_DATA)**2))

    def save(self):
        # put this here for now just to remember that it is important to save the reservoir
        #  state as well
        to_save = ("W_in, W_rs, W_out, res_state", self.W_in, self.W_reservoir, self.W_out, self.reservoir_state)
