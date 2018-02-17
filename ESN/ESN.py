import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class ESN():

    def __init__(self, input_size, output_size, reservoir_size=100, echo_param=0.6, 
                spectral_scale=1.0, init_echo_timesteps=100,
                regulariser=1e-8, input_weights_scale=(1/100.),
                debug_mode=False):

        # ARCHITECTURE PARAMS
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.activation_function = np.tanh
        self.input_weights_scale = input_weights_scale

        # RESOVOIR PARAMS
        self.spectral_scale = spectral_scale
        self.reservoir_state = np.zeros((1, self.reservoir_size))
        self.echo_param = echo_param
        self.init_echo_timesteps = init_echo_timesteps # number of inititial runs before training
        self.regulariser = regulariser

        # WEIGHTS
        #self.W_in = (np.random.randn(input_size, reservoir_size) - 0.5)*(1/1000.)
        #self.W_in = ((np.random.rand(input_size, reservoir_size) > 0.5).astype(int) - 0.5) *self.input_weights_scale 
        self.W_in = (np.random.rand(input_size, reservoir_size) - 0.5) *self.input_weights_scale 

        self.W_reservoir = []
        # self.__reservoir_norm_spectral_radius_norm_weights__()
        self.__reservoir_norm_spectral_radius_uniform_weights__()

        #self.W_reservoir = np.random.rand(self.reservoir_size, self.reservoir_size)-0.5

        self.W_out = []

        self.debug = debug_mode

        if self.debug: print("W_in: {}".format(self.W_in[:10]))
        if self.debug: print("W_res: {}".format(self.W_reservoir))

        # SOME EXTA STORE DATA
        self.training_signals = [] # reservoir state over time during training


    def copy(self):
        return ESN(self.input_size, self.output_size, self.reservoir_size, self.echo_param, 
                    self.spectral_scale, self.init_echo_timesteps,
                    self.regulariser, self.input_weights_scale, self.debug)

    def reset_reservoir(self):
        self.reservoir_state = np.zeros((1, self.reservoir_size))

    def __reservoir_norm_spectral_radius_norm_weights__(self):
        return self.__reservoir_norm_spectral_radius__(np.random.randn)

    def __reservoir_norm_spectral_radius_uniform_weights__(self):
        return self.__reservoir_norm_spectral_radius__(np.random.rand)

    def __reservoir_norm_spectral_radius_binary_weights__(self):
        def binary_distr(d0, d1):
            return (np.random.rand(d0, d1) + 0.5).astype(int)
        return self.__reservoir_norm_spectral_radius__(binary_distr)

    def __reservoir_norm_spectral_radius__(self, weight_distribution_function, offset=0.5):
        # self.W_reservoir = np.random.rand(reservoir_size, reservoir_size)
        self.W_reservoir = weight_distribution_function(self.reservoir_size, self.reservoir_size) - offset
        # make the spectral radius < 1 by dividing by the absolute value of the largest eigenvalue.
        self.W_reservoir /= max(abs(np.linalg.eig(self.W_reservoir)[0]))
        self.W_reservoir *= self.spectral_scale

    def __forward_to_res__(self, x_in):
        assert np.shape(x_in)[1] == np.shape(self.W_in)[0], "input of {} does not match input weights of {}".format(np.shape(x_in)[1], np.shape(self.W_in)[0])

        in_to_res = np.dot(x_in, self.W_in)
        res_to_res = np.dot(self.reservoir_state, self.W_reservoir)

        assert np.shape(in_to_res) == np.shape(res_to_res), "in-to-res input is {} whereas res-to-res input is {}".format(np.shape(in_to_res), np.shape(res_to_res))

        self.reservoir_state = (1.0 - self.echo_param)*self.reservoir_state + self.echo_param*self.activation_function(in_to_res + res_to_res)
        
        #res_to_out = np.dot(self.reservoir_state, self.W_out)
        return self.reservoir_state.flatten()

    def forward_to_out(self, x_in):
        assert len(self.W_out) > 0, "ESN has not been trained yet!"
        assert len(np.shape(x_in)) == 1, "input should have only 1 dimension. Dimension is: {}".format(np.shape(x_in))

        # print(np.shape(x_in))
        res_out = np.array(self.__forward_to_res__(np.array([x_in])))
        res_out = np.hstack((res_out, x_in)) # augment the data with the reservoir data
        # print(np.shape(res_out))
        assert np.shape(res_out)[0] == np.shape(self.W_out)[0], "res output is {}, whereas expected weights are {}".format(np.shape(res_out), np.shape(self.W_out))

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

    def generate(self, data, sample_step=None, plot=True, show_error=True):
        """ Pass the trained model. """
        # reset the reservoir
        #self.reset_reservoir()
        print(data)

        input_size = self.input_size-1 # -1 because of bias

        generated_data = []
        for i in range(0, len(data)-input_size):
            # run after the reservoir has "warmed-up"
            if i >= 1: #self.init_echo_timesteps:
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
        error_mean = np.mean((np.array(generated_data)[:, None] - data[(1+input_size):])**2)
        error_var = np.var((np.array(generated_data)[:, None] - data[(1+input_size):]))
        error = np.sqrt(error_mean/error_var)
        print('MSE generating test: %.7f' % error)
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

    def save(self):
        # put this here for now just to remember that it is important to save the reservoir
        #  state as well
        to_save = ("W_in, W_rs, W_out, res_state", self.W_in, self.W_reservoir, self.W_out, self.reservoir_state)
