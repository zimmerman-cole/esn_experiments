import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, ESN
from MackeyGlass.MackeyGlassGenerator import run
from Helper.utils import nrmse

if __name__ == '__main__':
    data = np.array([run(21100)]).reshape(-1, 1)
    data_mean = np.mean(data, axis=0)
    split = 20100
    X_train = np.array(data[:split-1])
    y_train = np.array(data[1:split])
    X_valid = np.array(data[split-1:-1])
    y_valid = np.array(data[split:])

    esn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    esn.initialize_input_weights(scale=1.0)
    esn.initialize_reservoir_weights(spectral_scale=1.25)

    lcesn = LCESN(1, 1, 5, echo_params=np.linspace(0.85, 0.5, 5), regulariser=1e-6, debug=True)
    lcesn.initialize_input_weights(scales=0.2)
    lcesn.initialize_reservoir_weights(spectral_scales=1.0)

    esn.train(X_train, y_train)
    lcesn.train(X_train, y_train)

    esn_outputs = []
    lcesn_outputs = []

    # GENERATIVE =================================================
    u_n_ESN = data[split]
    u_n_LCESN = data[split]
    for _ in range(len(data[split:])):
        u_n_ESN = esn.forward(u_n_ESN)
        esn_outputs.append(u_n_ESN)
        u_n_LCESN = lcesn.forward(u_n_LCESN)
        lcesn_outputs.append(u_n_LCESN)

    # SUPERVISED ====
    # for u_n in X_valid:
    #     esn_outputs.append(esn.forward(u_n))
    #     lcesn_outputs.append(lcesn.forward(u_n))
    # ============================================================

    esn_outputs = np.array(esn_outputs).squeeze()
    lcesn_outputs = np.array(lcesn_outputs).squeeze()

    print('  ESN MSE: %f' % nrmse(y_valid, esn_outputs, data_mean))
    print('LCESN MSE: %f' % nrmse(y_valid, lcesn_outputs, data_mean))

    if 1:
        f, ax = plt.subplots(figsize=(12, 12))
        ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
        #ax.plot(range(len(lcesn_outputs)), lcesn_outputs, label='LCESN')
        ax.plot(range(len(y_valid)), y_valid, label='True')
        plt.legend()

        for res in lcesn.reservoirs:
            all_signals = np.array(res.signals).squeeze()[:100, :5].T

            f, ax = plt.subplots()
            ax.set_title(u'Reservoir %d. \u03B5: %.2f' % (res.idx, res.echo_param))
            for signals in all_signals:
                ax.plot(range(len(signals)), signals)

            plt.show()
