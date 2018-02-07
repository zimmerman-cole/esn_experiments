import numpy as np
import pickle as pkl

class ESN():

    def __init__(self, input_size, output_size, reservoir_size=100, echo_param=0.6, spectral_scale=1.0, init_echo_timesteps=100):

        # ARCHITECTURE PARAMS
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.activation_function = np.tanh

        # RESOVOIR PARAMS
        self.spectral_scale = spectral_scale
        self.reservoir_state = np.zeros((reservoir_size, 1))
        self.echo_param = echo_param
        self.init_echo_timesteps = init_echo_timesteps # number of inititial runs before training

        # WEIGHTS
        self.W_in = np.random.randn(input_size, reservoir_size)

        self.W_reservoir = []
        self.__reservoir_norm_spectral_radius_norm_weights__()

        self.W_out = []


    def copy(self):
        return ESN(self.input_size, self.output_size, self.reservoir_size, self.echo_param, self.spectral_scale, self.init_echo_timesteps)

    def __reservoir_norm_spectral_radius_norm_weights__(self):
        return self.__reservoir_norm_spectral_radius__(np.random.randn)

    def __reservoir_norm_spectral_radius_binary_weights__(self):
        def binary_distr(d0, d1):
            return (np.random.rand(d0, d1) + 0.5).astype(int)
        return self.__reservoir_norm_spectral_radius__(binary_distr)

    def __reservoir_norm_spectral_radius__(self, weight_distribution_function):
        # self.W_reservoir = np.random.rand(reservoir_size, reservoir_size)
        self.W_reservoir = weight_distribution_function(self.reservoir_size, self.reservoir_size)
        # make the spectral radius < 1 by dividing by the absolute value of the largest eigenvalue.
        self.W_reservoir /= abs(max(np.linalg.eig(self.W_reservoir)[0]))
        self.W_reservoir *= self.spectral_scale

    def __forward_to_res__(self, x_in):
        # print(x_in)
        in_to_res = np.dot(x_in, self.W_in).T
        res_to_res = np.dot(self.W_reservoir, self.reservoir_state)
        self.reservoir_state = (
            (1.0 - self.echo_param)*self.reservoir_state +
            self.echo_param*self.activation_function(in_to_res + res_to_res)
        )
        #res_to_out = np.dot(self.reservoir_state, self.W_out)
        return self.reservoir_state.flatten()

    def forward_to_out(self, x_in):
        assert len(self.W_out) > 0, "ESN has not been trained yet!"

        res_out = self.__forward_to_res__(x_in)
        res_out = np.hstack((res_out, x_in, 1)) # augment the data with the reservoir data
        res_to_out = np.dot(res_out, self.W_out)

        return res_to_out

    def train(self, data):
        # first we run the ESN for a few inputs so that the reservoir starts echoing
        data_init = data[:self.init_echo_timesteps]
        data_train = data[self.init_echo_timesteps:]
        for d in data_init:
            # print(d)
            _ = self.__forward_to_res__(d)
        print("-"*10+"INITIAL ECHO TIMESTEPS DONE."+"-"*10)

        # now train the reservoir after we have set up the echo state
        y_out = np.zeros((np.shape(data_train)[0], self.reservoir_size+self.input_size+1))
        for idx,d in enumerate(data_train):
            y = self.__forward_to_res__(d)
            y = np.hstack((y, d, 1)) # augment the data with the reservoir data
            y_out[idx, :] = y
        print("-"*10+"DATA PUT THROUGH RESERVOIR DONE."+"-"*10)

        # do linear regression between the inputs and the output
        X_train = y_out[:-1]
        y_target = data_train[1:]
        # print(np.shape(X_train))
        # print(np.shape(y_target))
        lsq_result = np.linalg.lstsq(X_train, y_target)
        self.W_out = lsq_result[0]

        print("-"*10+"LINEAR REGRESSION ON OUTPUT DONE."+"-"*10)
        print("ESN trained!")

    def predict(self, data):
        # We do not need to 'initialise' the ESN because the training phase already did this
        y_out = np.zeros((np.shape(data)[0], 1))
        for idx,d in enumerate(data):
            y = self.forward_to_out(d)
            y_out[idx, :] = y

        return y_out

    def mean_l2_error(self, y_out, y_pred):
        return np.mean(abs(np.array(y_out) - np.array(y_pred)))

    def save(self):
        # put this here for now just to remember that it is important to save the reservoir
        #  state as well
        to_save = ("W_in, W_rs, W_out, res_state", self.W_in, self.W_reservoir, self.W_out, self.reservoir_state)
