from ESN.ESN import ESN2 as ESN
from MackeyGlass.MackeyGlassGenerator import run, onExit

import numpy as np
import matplotlib.pyplot as plt

# because ESN are stochastic, this will train a ESN multiple times and compute the average MSE
def ESN_stochastic_train(data, train_split, val_split, esn, num_runs, MEAN_OF_DATA, seed=None):

    data_train = data[:train_split]
    #print(np.shape(data_train))
    input_size = esn.input_size - 1
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

    data_test = data[train_split:val_split]
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

    data_val = data[val_split:]

    # check we have made the data correctly
   # assert data_test_X[24, 0] == data_test_yoffsets23], "teblock_offsets data is not correctly made!"

    nmse_test = []
    nmse_train = []
    gen_test = []
    gen_train = []
    gen_val = []
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

        mse_test = esn_copy.nmse(data_test_y, y_pred_test, MEAN_OF_DATA)
        mse_train = esn_copy.nmse(data_train_y[esn.init_echo_timesteps:], y_pred_train, MEAN_OF_DATA)
        #mean_mse_test += mse_test
        #mean_mse_train += mse_train
        nmse_test.append(mse_test)
        nmse_train.append(mse_train)

    #gen_err, generated_data = esn_copy.generate(data_test[:1000], plot=False)
        gen_err_test, generated_data_test = esn_copy.generate(data_test[:2000], MEAN_OF_DATA, plot=False, show_error=False)
        gen_err_train, generated_data_train = esn_copy.generate(data_train[:2000], MEAN_OF_DATA, plot=False, show_error=False)

        gen_err_val, generated_data_val = esn_copy.generate(data_val, MEAN_OF_DATA, plot=False, show_error=False)

        #mean_gen_test += gen_err_test
        #mean_gen_train += gen_err_train
        gen_test.append(gen_erMEAN_OF_DATA = np.mean(data)r_test)
        gen_train.append(gen_err_train)
        gen_val.append(gen_err_val)

        print("iter: {} -- NRMSE (SUP) Error TEST: {}, TRAIN: {}".format(e, mse_test, mse_train))
        print("iter: {} -- NRMSE (GEN) Error TEST: {}, TRAIN: {}".format(e, gen_err_test, gen_err_train))
        print("iter: {} -- NRMSE (GEN) Error VAL: {}".format(e, gen_err_val))

    #g_data = [generated_data, generated_data, generated_data, generated_data]
    #d = data_test[(esn_copy.init_echo_timesteps+esn_copy.input_size-1):1000, 0]
    #a_data = [d, d, d, d]
    #fancy_plot(g_data, a_data, 2, 2)

    #mean_mse_test /= num_runs
    #mean_mse_train /= num_runs
    #mean_gen_train /= num_runs
    #mean_gen_test /= num_runs
    mean_mse_test = np.mean(nmse_test)
    std_mse_test = np.std(nmse_test) / np.sqrt(len(nmse_test))

    mean_mse_train = np.mean(nmse_train)
    std_mse_train = np.std(nmse_train) / np.sqrt(len(nmse_train))

    mean_gen_test = np.mean(gen_test)
    std_gen_test = np.std(gen_test) / np.sqrt(len(gen_test))

    mean_gen_train = np.mean(gen_train)
    std_gen_train = np.std(gen_train) / np.sqrt(len(gen_train))

    mean_gen_val = np.mean(gen_val)
    std_gen_val = np.std(gen_val) / np.sqrt(len(gen_val))

    print("\n\nFINAL -- Supervised Mean L2 Error TEST: {} +- {:.6f}, TRAIN: {} +- {:.6f}".format(
            mean_mse_test, std_mse_test, mean_mse_train, std_mse_train))
    print("\n\nFINAL -- Generative Mean L2 Error TEST: {} +- {:.6f}, TRAIN: {} +- {:.6f}".format(
            mean_gen_test, std_gen_test, mean_gen_train, std_gen_train))
    print("\n\nFINAL -- Generative Mean L2 Error VAL: {} +- {:.6f}".format(
            mean_gen_val, std_gen_val))

    return gen_err_test, generated_data_test, data_test[(esn_copy.init_echo_timesteps+esn.input_size-1):2000,0], esn_copy.training_signals

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
    data = np.array([run(21100)]).T
    # data_train_test = data[:20000]
    # data_val = data[20000:]
    #data = np.loadtxt('../../../MackeyGlass_t17.txt')
    #data -= np.mean(data)
    #data += 100.0
    #data -= np.mean(data)
    #print(np.std(data))
    #data /= np.std(data)
    #data *= 2.0
    # onExit(data)
    #esn = ESN(input_size=2, output_size=1, reservoir_size=1000, echo_param=0.1, spectral_scale=1.1, init_echo_timesteps=100, regulariser=1e-0, debug_mode=True)
    #ESN_stochastic_train(data, 7000, esn, 1)
    MEAN_OF_DATA = np.mean(data)
    print("DATA MEAN: {}".format(MEAN_OF_DATA))
    #np.random.seed(42)
    g_data = []
    a_data = []
    titles = []
    training_signals = []
    # run a few test of different hyperparameters
    count = 0
    for e in [0.85]:
        for r in [1000]:
            for reg in [1e-6]:
                for s in [1.25]:
                    for a in [1/1.]:
                        print("EXPERIMENT - ECHO {}, RES {}, REG {}, SPECT {}, W_IN {}".format(e, r, reg, s, a))
                        esn = ESN(input_size=2, output_size=1, reservoir_size=r, echo_param=e, spectral_scale=s, 
                                    init_echo_timesteps=100,regulariser=reg, input_weights_scale=a, debug_mode=False)
                        gen_err, g_, a_, signals_ = ESN_stochastic_train(data, 14000, 20000, esn, 1, MEAN_OF_DATA)
                        g_data.append(g_)
                        a_data.append(a_)
                        titles.append(("ECHO: {:.2f}, SPEC: {:.2f}, REG: {}, W-IN: {}".format(e, s, reg, a)))
                        training_signals.append(signals_[:1000, :10])
                        count += 1
                        print("COUNT: {}".format(count))

                        # f, ax = plt.subplots()
                        # ax.plot(range(len(g_)),  a_, label='True data', c='red')
                        # ax.scatter(range(len(g_)),  a_, s=4.5, c='black', alpha=0.5) 
                        # ax.plot(range(len(g_)),  g_, label='Generated data', c='blue')
                        # ax.scatter(range(len(g_)),  g_, s=4.5, c='black', alpha=0.5)

                        # residual = a_ - g_
                        # ax.plot(range(len(residual)), residual, c='green', label='error')
                        # ax.plot(range(len(residual)), np.zeros(len(residual)), c='r', linestyle='--')

                        # plt.legend()
                        # plt.show()



    # plot the generated data versus actual data of the ESNs
    fancy_plot_generative(g_data, a_data, 3, 3, titles=titles)

    # plot some signals of the ESNs
    fancy_plot_signals(training_signals, 3, 3, titles=titles)

    plt.show()







