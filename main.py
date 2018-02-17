from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run, onExit

import numpy as np
import matplotlib.pyplot as plt

# because ESN are stochastic, this will train a ESN multiple times and compute the average MSE
def ESN_stochastic_train(data, train_split, esn, num_runs, seed=None):

    data_train = data[:train_split]
    #print(np.shape(data_train))
    input_size = 1
    data_train_X = np.zeros((np.shape(data_train)[0] - input_size, input_size))
    data_train_y = np.zeros((np.shape(data_train)[0] - input_size, 1))
    #print(np.shape(data_train_X))
    #print(np.shape(data_train_y))
    for i in range(np.shape(data_train)[0] - input_size):
        data_train_X[i, :] = data_train[i:(i+input_size), 0]
        data_train_y[i, :] = data_train[i+input_size]

    #data_train_X = np.hstack((data_train[:-1], np.ones((np.shape(data_train)[0]-1, 1))))
    #data_train_y = data_train[1:]
    data_train_X = np.hstack((data_train_X, np.ones((np.shape(data_train_X)[0], 1))))

    #print(data_train_X)
    
    # check we have made the data correctly
   # assert data_train_X[13, 0] == data_train_y[12], "training data is not correctly made!"

    data_test = data[train_split:]
    #data_test_X = np.hstack((data_test[:-1], np.ones((np.shape(data_test)[0]-1, 1))))
    #data_test_y = data_test[1:]

    data_test_X = np.zeros((np.shape(data_test)[0] - input_size, input_size))
    data_test_y = np.zeros((np.shape(data_test)[0] - input_size, 1))
    #print(np.shape(data_test_X))
    #print(np.shape(data_test_y))
    for i in range(np.shape(data_test)[0] - input_size):
        data_test_X[i, :] = data_test[i:(i+input_size), 0]
        data_test_y[i, :] = data_test[i+input_size]

    data_test_X = np.hstack((data_test_X, np.ones((np.shape(data_test_X)[0], 1))))

    # check we have made the data correctly
   # assert data_test_X[24, 0] == data_test_yoffsets23], "teblock_offsets data is not correctly made!"

    mean_mse_test = 0
    mean_mse_train = 0
    #for e in range(num_runs):
    #esn_copy = esn.copy()
    esn.train(data_train_X, data_train_y)

    if esn.debug:
        plt.bar(range(esn.input_size+esn.reservoir_size), esn.W_out)
        plt.title("Output Weights")
        plt.show()

    #y_pred_test = esn.predict(data_test_X)
    # plt.plot(range(len(y_pred_test)), y_pred_test)
    # plt.plot(range(len(data_test_X)), data_test_X)
    # plt.show()
    #y_pred_train = esn.predict(data_train_X, reset_res=True)

    #mse_test = esn.mean_l2_error(data_test_y, y_pred_test)
    #mse_train = esn.mean_l2_error(data_train_y[esn.init_echo_timesteps:], y_pred_train)
    #mean_mse_test += mse_test
    #mean_mse_train += mse_train
    #print("iter: {} -- Mean L2 Error TEST: {}, TRAIN: {}".format(e, mse_test, mse_train))
    # print("iter: {} -- Mean L2 Error TEST: {}".format(e, mse_test))

    #gen_err, generated_data = esn_copy.generate(data_test[:1000], plot=False)
    gen_err, generated_data = esn.generate(data_test[:1500], plot=False)

    #g_data = [generated_data, generated_data, generated_data, generated_data]
    #d = data_test[(esn_copy.init_echo_timesteps+esn_copy.input_size-1):1000, 0]
    #a_data = [d, d, d, d]
    #fancy_plot(g_data, a_data, 2, 2)

    mean_mse_test /= num_runs
    mean_mse_train /= num_runs
    print("\n\nFINAL -- Mean L2 Error TEST: {}, TRAIN: {}".format(mean_mse_test, mean_mse_train))

    return gen_err, generated_data, data_test[(1+esn.input_size-1):1500,0], esn.training_signals

def fancy_plot_generative(generated_data, actual_data, num_rows, num_cols, titles=[]):
    # generated data and actual data should be lists of data sets so that subplots can be made
    assert type(generated_data) == type([]), "data must be a list of numpys"
    assert type(actual_data) == type([]), "data must be a list of numpys"
    f, ax = plt.subplots(num_rows, num_cols, sharex=False, sharey=False)

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

def fancy_plot_signals(signals_data, num_rows, num_cols, titles=[]):
    assert type(signals_data) == type([]), "data must be a list of numpys"
    f, ax = plt.subplots(num_rows, num_cols, sharex=False, sharey=False)

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r*num_cols + c
            s_data = signals_data[idx]
            ax[r,c].plot(s_data)

            if len(titles) > 0:
                ax[r, c].set_title(titles[idx])

    #plt.show(block=False)
    plt.draw()


if __name__ == "__main__":
    data = np.array([run(4000)]).T
    #data = np.loadtxt('../../../MackeyGlass_t17.txt')
    #data -= np.mean(data)
    #data += 100.0
    #data -= np.mean(data)
    #print(np.std(data))
    #data /= np.std(data)
    #onExit(data)
    #esn = ESN(input_size=2, output_size=1, reservoir_size=1000, echo_param=0.1, spectral_scale=1.1, init_echo_timesteps=100, regulariser=1e-0, debug_mode=True)
    #ESN_stochastic_train(data, 7000, esn, 1)

    g_data = []
    a_data = []
    titles = []
    training_signals = []
    # run a few test of different hyperparameters
    count = 0
    for e in [0.5, 0.5, 0.6]:
        for r in [1000]:
            for reg in [1e-5]:
                for s in [1.2, 1.2, 1.3]:
                    for a in [1/1000.]:
                        print("EXPERIMENT - ECHO {}, RES {}, REG {}, SPECT {}, W_IN {}".format(e, r, reg, s, a))
                        esn = ESN(input_size=2, output_size=1, reservoir_size=r, echo_param=e, spectral_scale=s, init_echo_timesteps=100,regulariser=reg, input_weights_scale=a, debug_mode=False)
                        gen_err, g_, a_, signals_ = ESN_stochastic_train(data, 2000, esn, 1)
                        g_data.append(g_)
                        a_data.append(a_)
                        titles.append(("ECHO: {:.2f}, SPEC: {:.2f}, REG: {}, W-IN: {}".format(e, s, reg, a)))
                        training_signals.append(signals_[:1000, :10])
                        count += 1
                        print("COUNT: {}".format(count))


    # plot the generated data versus actual data of the ESNs
    fancy_plot_generative(g_data, a_data, 3, 3, titles=titles)

    # plot some signals of the ESNs
    fancy_plot_signals(training_signals, 3, 3, titles=titles)

    plt.show()







