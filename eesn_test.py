import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, EESN, ESN
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

    # esn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    # esn.initialize_input_weights(scale=1.0)
    # esn.initialize_reservoir_weights(spectral_scale=1.25)

    eesn = EESN(1, 1, 10, 
                reservoir_sizes=np.linspace(10, 500, 10, endpoint=True).astype(int), 
                echo_params=np.linspace(0.1, 0.8, 10, endpoint=True), 
                regulariser=1e-4, debug=True,
                init_echo_timesteps=300)
    eesn.initialize_input_weights(scales=1.0, strategies='uniform')
    eesn.initialize_reservoir_weights(
                spectral_scales=np.linspace(1.0, 1.35, 10, endpoint=True).tolist(),
                strategies=['uniform']*10
                )

    # esn.train(X_train, y_train)
    eesn.train(X_train, y_train)

    # esn_outputs = []
    eesn_outputs = []

    # GENERATIVE =================================================
    # u_n_ESN = data[split]
    # print(X_train[-10:])
    # print(data[split-2])
    # print(data[split-1])
    # print(data[split])
    u_n_EESN = data[split-1]
    for _ in range(len(data[split:])):
        # u_n_ESN = esn.forward(u_n_ESN)
        # esn_outputs.append(u_n_ESN)
        u_n_EESN = eesn.forward(u_n_EESN)
        eesn_outputs.append(u_n_EESN)

    # SUPERVISED ====
    # for u_n in X_valid:
    #     esn_outputs.append(esn.forward(u_n))
    #     lcesn_outputs.append(lcesn.forward(u_n))
    # ============================================================

    # esn_outputs = np.array(esn_outputs).squeeze()
    eesn_outputs = np.array(eesn_outputs).squeeze()

    # print('  ESN MSE: %f' % mse(y_valid, esn_outputs))
    print('EESN MSE: %f' % mse(y_valid, eesn_outputs))

    if 1:
        f, ax = plt.subplots(figsize=(12, 12))
        # ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
        ax.plot(range(len(eesn_outputs)), eesn_outputs, label='EESN')
        ax.plot(range(len(y_valid)), y_valid, label='True')
        plt.legend()

        for res in eesn.reservoirs:
            all_signals = np.array(res.signals).squeeze()[:1000, :5].T

            f, ax = plt.subplots()
            ax.set_title(u'Reservoir %d. \u03B5: %.2f' % (res.idx, res.echo_param))
            for signals in all_signals:
                ax.plot(range(len(signals)), signals)

            plt.show()