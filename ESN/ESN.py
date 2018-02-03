import numpy as np
import pickle as pkl

class ESN():

    def __init__(self, input_size, output_size, reservoir_size=100):

        # WEIGHTS

        # for now we just ue randn initialisation
        self.W_in = np.random.randn(input_size, reservoir_size)
        self.W_reservoir = np.random.rand(reservoir_size, reservoir_size) - 0.5
        # TODO: use spectral radius to init resovoir weights
        self.W_out = []

        # RESOVOIR PARAMS
        self.reservoir_state = np.zeros((reservoir_size, 1))
        self.echo_param = 0.6
        self.init_echo_timesteps = 100 # number of inititial runs before training

        # ARCHITECTURE PARAMS

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.activation_function = np.tanh

    def __forward_to_res__(self, x_in):
        # print(x_in)
        in_to_res = np.dot(x_in, self.W_in)
        res_to_res = np.dot(self.reservoir_state.T, self.W_reservoir)
        self.reservoir_state = (
            self.echo_param*self.reservoir_state +
            (1.0 - self.echo_param)*self.activation_function(in_to_res + res_to_res)
        )
        #res_to_out = np.dot(self.reservoir_state, self.W_out)
        return self.reservoir_state.flatten()

    def forward_to_out(self, x_in):
        assert len(self.W_out) > 0, "ESN has not been trained yet!"

        res_out = self.__forward_to_res__(x_in)
        res_out = np.hstack((res_out, x_in)) # augment the data with the reservoir data
        res_to_out = np.dot(res_out, self.W_out)

        return res_to_out

    def train(self, data):
        # first we run the ESN for a few inputs so that the reservoir starts echoing
        data_init = data[:self.init_echo_timesteps]
        data_train = data[self.init_echo_timesteps:]
        for d in data_init:
            # print(d)
            _ = self.__forward_to_res__(d)

        # now train the reservoir after we have set up the echo state
        y_out = []
        for d in data_train:
            y = self.__forward_to_res__(d)
            y = np.hstack((y, d)) # augment the data with the reservoir data
            y_out.append(y)

        # do linear regression between the inputs and the output
        X_train = y_out[:-1]
        y_target = data_train[1:]
        lsq_result = np.linalg.lstsq(X_train, y_target)
        self.W_out = lsq_result[0]

        print("ESN trained!")

    def predict(self, data):
        # We do not need to 'initialise' the ESN because the trianing phase already did this
        y_out = []
        for d in data:
            y = self.forward_to_out(d)
            y_out.append(y)

        return y_out

    def mean_l1_error(self, y_out, y_pred):
        return np.mean(abs(np.array(y_out) - np.array(y_pred)))

    def save(self):
        # put this here for now just to remember that it is important to save the reservoir
        #  state as well
        to_save = ("W_in, W_rs, W_out, res_state", self.W_in, self.W_reservoir, self.W_out, self.reservoir_state)
