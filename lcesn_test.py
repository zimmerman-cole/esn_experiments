import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, ESN
from MackeyGlass.MackeyGlassGenerator import run


def mse(y1, y2):
    return np.mean((y1-y2)**2)


if __name__ == '__main__':
    data = np.array([run(21100)]).reshape(-1, 1)
    split = 20100
    X_train = data[:split-1]
    y_train = data[1:split]
    X_valid = data[split:-1]
    y_valid = data[split+1:]

    esn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    esn.initialize_input_weights(scale=1.0)
    esn.initialize_reservoir_weights(spectral_scale=1.25)

    lcesn = LCESN(1, 1, 5, echo_params=0.85, regulariser=1e-6)
    lcesn.initialize_input_weights(scales=1.0)
    lcesn.initialize_reservoir_weights(spectral_scales=1.25)

    esn.train(X_train, y_train)
    lcesn.train(X_train, y_train)

    esn_outputs = []
    lcesn_outputs = []
    for u_n in X_valid:
        esn_outputs.append(esn.forward(u_n))
        lcesn_outputs.append(lcesn.forward(u_n))

    esn_outputs = np.array(esn_outputs).squeeze()
    lcesn_outputs = np.array(lcesn_outputs).squeeze()

    print('  ESN MSE: %f' % mse(y_valid, esn_outputs))
    print('LCESN MSE: %f' % mse(y_valid, lcesn_outputs))

    if 0:
        f, axarr = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 12))
        axarr[0].plot(range(len(esn_outputs)), esn_outputs)
        axarr[0].set_title('ESN')
        axarr[1].plot(range(len(lcesn_outputs)), lcesn_outputs)
        axarr[1].set_title('LCESN')
        axarr[2].plot(range(len(y_valid)), y_valid)
        axarr[2].set_title('True')
        plt.show()

