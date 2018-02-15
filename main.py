from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run, onExit

import numpy as np
import matplotlib.pyplot as plt

# because ESN are stochastic, this will train a ESN multiple times and compute the average MSE
def ESN_stochastic_train(data, train_split, esn, num_runs, seed=None):

    data_train = data[:train_split]
    print(np.shape(data_train))
    input_size = 1
    data_train_X = np.zeros((np.shape(data_train)[0] - input_size, input_size))
    data_train_y = np.zeros((np.shape(data_train)[0] - input_size, 1))
    print(np.shape(data_train_X))
    print(np.shape(data_train_y))
    for i in range(np.shape(data_train)[0] - input_size):
        data_train_X[i, :] = data_train[i:(i+input_size), 0]
        data_train_y[i, :] = data_train[i+input_size]

    #data_train_X = np.hstack((data_train[:-1], np.ones((np.shape(data_train)[0]-1, 1))))
    #data_train_y = data_train[1:]
    data_train_X = np.hstack((data_train_X, np.ones((np.shape(data_train_X)[0], 1))))

    print(data_train_X)
    
    # check we have made the data correctly
   # assert data_train_X[13, 0] == data_train_y[12], "training data is not correctly made!"

    data_test = data[train_split:]
    #data_test_X = np.hstack((data_test[:-1], np.ones((np.shape(data_test)[0]-1, 1))))
    #data_test_y = data_test[1:]

    data_test_X = np.zeros((np.shape(data_test)[0] - input_size, input_size))
    data_test_y = np.zeros((np.shape(data_test)[0] - input_size, 1))
    print(np.shape(data_test_X))
    print(np.shape(data_test_y))
    for i in range(np.shape(data_test)[0] - input_size):
        data_test_X[i, :] = data_test[i:(i+input_size), 0]
        data_test_y[i, :] = data_test[i+input_size]

    data_test_X = np.hstack((data_test_X, np.ones((np.shape(data_test_X)[0], 1))))

    # check we have made the data correctly
   # assert data_test_X[24, 0] == data_test_yoffsets23], "teblock_offsets data is not correctly made!"

    mean_mse_test = 0
    mean_mse_train = 0
    for e in range(num_runs):
        esn_copy = esn.copy()
        esn_copy.train(data_train_X, data_train_y)

        if esn_copy.debug:
            plt.bar(range(esn_copy.input_size+esn.reservoir_size), esn_copy.W_out)
            plt.title("Output Weights")
            plt.show()

        y_pred_test = esn_copy.predict(data_test_X)
        # plt.plot(range(len(y_pred_test)), y_pred_test)
        # plt.plot(range(len(data_test_X)), data_test_X)
        # plt.show()
        y_pred_train = esn_copy.predict(data_train_X, reset_res=True)

        mse_test = esn_copy.mean_l2_error(data_test_y, y_pred_test)
        mse_train = esn_copy.mean_l2_error(data_train_y[esn_copy.init_echo_timesteps:], y_pred_train)
        mean_mse_test += mse_test
        mean_mse_train += mse_train
        print("iter: {} -- Mean L2 Error TEST: {}, TRAIN: {}".format(e, mse_test, mse_train))
        # print("iter: {} -- Mean L2 Error TEST: {}".format(e, mse_test))

        gen_err, generated_data = esn_copy.generate(data_test[:1000], plot=False)

    #g_data = [generated_data, generated_data, generated_data, generated_data]
    #d = data_test[(esn_copy.init_echo_timesteps+esn_copy.input_size-1):1000, 0]
    #a_data = [d, d, d, d]
    #fancy_plot(g_data, a_data, 2, 2)

    mean_mse_test /= num_runs
    mean_mse_train /= num_runs
    print("\n\nFINAL -- Mean L2 Error TEST: {}, TRAIN: {}".format(mean_mse_test, mean_mse_train))

    return gen_err, generated_data, data_test[(esn_copy.init_echo_timesteps+esn_copy.input_size-1):1000,0]

def fancy_plot(generated_data, actual_data, num_rows, num_cols, titles=[]):
    # generated data and actual data should be lists of data sets so that subplots can be made
    f, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r*num_cols + c
            g_data = generated_data[idx]
            a_data = actual_data[idx]
            ax[r,c].plot(range(len(g_data)), a_data, label='True data', c='red')
            ax[r,c].scatter(range(len(g_data)), a_data, s=4.5, c='black', alpha=0.5) 
            ax[r,c].plot(range(len(g_data)), g_data, label='Generated data', c='blue')
            ax[r,c].scatter(range(len(g_data)), g_data, s=4.5, c='black', alpha=0.5)

            if len(titles) > 0:
                ax[r, c].set_title(titles[idx])

    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = np.array([run(12000)]).T
    data -= np.mean(data)
    print(np.std(data))
    #data /= np.std(data)
    #onExit(data)
    #esn = ESN(input_size=2, output_size=1, reservoir_size=1000, echo_param=0.1, spectral_scale=1.1, init_echo_timesteps=100, regulariser=1e-0, debug_mode=True)
    #ESN_stochastic_train(data, 7000, esn, 1)

    g_data = []
    a_data = []
    titles = []
    # run a few test of different hyperparameters
    for e in np.linspace(0, 1, 16):
        esn = ESN(input_size=2, output_size=1, reservoir_size=1000, echo_param=e, spectral_scale=1.1, init_echo_timesteps=100, regulariser=1e-0, debug_mode=False)
        gen_err, g_, a_ = ESN_stochastic_train(data, 7000, esn, 1)
        g_data.append(g_)
        a_data.append(a_)
        titles.append(("ECHO PARAM: {}:".format(e)))

    fancy_plot(g_data, a_data, 4, 4, titles=titles)







