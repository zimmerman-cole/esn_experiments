from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run

# because ESN are stochastic, this will train a ESN multiple times and compute the average MSE
def ESN_stochastic_train(data, train_split, esn, num_runs):

    data_train = data[:train_split]
    data_test = data[train_split:]

    mean_mse = 0
    for e in range(num_runs):
        esn_copy = esn.copy()
        esn_copy.train(data_train)

        y_pred = esn_copy.predict(data_test)

        mse = esn_copy.mean_l2_error(data_test[:-1], y_pred[1:])
        mean_mse += mse
        print("iter: {} -- Mean L2 Error: {}".format(e, mse))

    mean_mse /= num_runs
    print("\n\nFINAL -- Mean L2 Error: {}".format(mean_mse))


if __name__ == "__main__":
    data = run(5000)
    esn = ESN(input_size=1, output_size=1, reservoir_size=1000, echo_param=0.3, spectral_scale=1.25, init_echo_timesteps=100)
    ESN_stochastic_train(data, 3000, esn, 1)
