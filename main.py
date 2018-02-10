from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run

import numpy as np

# because ESN are stochastic, this will train a ESN multiple times and compute the average MSE
def ESN_stochastic_train(data, train_split, esn, num_runs, seed=None):

    data_train = data[:train_split]
    print(np.shape(data_train))
    data_train_X = np.hstack((data_train[:-1], np.ones((np.shape(data_train)[0]-1, 1))))
    data_train_y = data_train[1:]

    print(data_train_X)
    
    # check we have made the data correctly
    assert data_train_X[13, 0] == data_train_y[12], "training data is not correctly made!"

    data_test = data[train_split:-500]
    data_test_X = np.hstack((data_test[:-1], np.ones((np.shape(data_test)[0]-1, 1))))
    data_test_y = data_test[1:]

    # check we have made the data correctly
    assert data_test_X[24, 0] == data_test_y[23], "testing data is not correctly made!"

    mean_mse_test = 0
    mean_mse_train = 0
    for e in range(num_runs):
        esn_copy = esn.copy()
        esn_copy.train(data_train_X, data_train_y)

        y_pred_test = esn_copy.predict(data_test_X)
        y_pred_train = esn_copy.predict(data_train_X, reset_res=True)

        mse_test = esn_copy.mean_l2_error(data_test_y, y_pred_test)
        mse_train = esn_copy.mean_l2_error(data_train_y[esn_copy.init_echo_timesteps:], y_pred_train)
        mean_mse_test += mse_test
        mean_mse_train += mse_train
        print("iter: {} -- Mean L2 Error TEST: {}, TRAIN: {}".format(e, mse_test, mse_train))

        esn_copy.generate(data_train[:1000])

    mean_mse_test /= num_runs
    mean_mse_train /= num_runs
    print("\n\nFINAL -- Mean L2 Error TEST: {}, TRAIN: {}".format(mean_mse_test, mean_mse_train))


if __name__ == "__main__":
    data = np.array([run(12000)]).T
    esn = ESN(input_size=2, output_size=1, reservoir_size=1000, echo_param=0.26, spectral_scale=0.05, init_echo_timesteps=500)
    ESN_stochastic_train(data, 7000, esn, 1)
