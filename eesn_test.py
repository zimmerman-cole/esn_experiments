import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, EESN, ESN
from MackeyGlass.MackeyGlassGenerator import run

from Helper.utils import nrmse

# def mse(y1, y2):
#     return np.mean((y1-y2)**2)

# def nrmse(y_true, y_pred, MEAN_OF_DATA):
#     return np.sqrt(np.sum((y_true - y_pred)**2)/np.sum((y_true - MEAN_OF_DATA)**2))


if __name__ == '__main__':
    data = np.array([run(21100)]).reshape(-1, 1)
    MEAN_OF_DATA = np.mean(data)
    split = 20100
    X_train = data[:split-1]
    y_train = data[1:split]
    X_valid = data[split-1:-1]
    y_valid = data[split:]

    # esn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    # esn.initialize_input_weights(scale=1.0)
    # esn.initialize_reservoir_weights(spectral_scale=1.25)

    # esn.train(X_train, y_train)

    res_ranges = [(10, 500), (300, 300), (200, 400)]
    echo_ranges = [(0.1, 0.8), (0.8, 0.1), (0.8, 0.8), (0.3, 0.5)]
    spect_ranges = [(1.0, 1.35), (1.35, 1.0), (1.25, 1.25), (1.0, 1.0)]
    num_res = 10
    
    for r in res_ranges:
        for e in echo_ranges:
            for s in spect_ranges:
                _reservoirs = np.round(np.linspace(r[0], r[1], num_res, endpoint=True).astype(int), 3)
                _echoes = np.round(np.linspace(e[0], e[1], num_res, endpoint=True), 3)
                _spectrals = np.round(np.linspace(s[0], s[1], num_res, endpoint=True), 3).tolist()
                print("EXPERIMENT: \n\tRES: {}, \n\tECH: {}, \n\tSPEC: {}, ".format(_reservoirs, _echoes, _spectrals))
                eesn = EESN(1, 1, 10, 
                            reservoir_sizes=_reservoirs, 
                            echo_params=_echoes, 
                            regulariser=1e-4, debug=True,
                            init_echo_timesteps=300)
                eesn.initialize_input_weights(scales=1.0, strategies='uniform')
                eesn.initialize_reservoir_weights(
                            spectral_scales=_spectrals,
                            strategies=['uniform']*num_res
                            )
                eesn.train(X_train, y_train)

    eesn_outputs = []

    # GENERATIVE =================================================
    u_n_EESN = data[split-1]
    for _ in range(len(data[split:])):
        u_n_EESN = esn.forward(u_n_EESN)
        eesn_outputs.append(u_n_EESN)

    # SUPERVISED ====
    # for u_n in X_valid:
    #     esn_outputs.append(esn.forward(u_n))
    #     lcesn_outputs.append(lcesn.forward(u_n))
    # ============================================================

    # esn_outputs = np.array(esn_outputs).squeeze()
    eesn_outputs = np.array(eesn_outputs).squeeze()
    y_vals = y_valid.squeeze()
    print(np.shape(eesn_outputs))
    print(np.shape(y_vals))
    print(np.vstack((eesn_outputs, y_vals)))

    # print('  ESN MSE: %f' % mse(y_valid, esn_outputs))
    # print('EESN MSE: %f' % mse(y_valid, eesn_outputs))
    print('EESN NRMSE: %f' % nrmse(y_vals, eesn_outputs, MEAN_OF_DATA))

    if 0:
        f, ax = plt.subplots(figsize=(12, 12))
        # ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
        ax.plot(range(len(eesn_outputs)), eesn_outputs, label='EESN')
        ax.plot(range(len(y_valid)), y_valid, label='True')
        plt.legend()

        # for res in esn.reservoirs:
        #     all_signals = np.array(res.signals).squeeze()[:1000, :5].T

        #     f, ax = plt.subplots()
        #     ax.set_title(u'Reservoir %d. \u03B5: %.2f' % (res.idx, res.echo_param))
        #     for signals in all_signals:
        #         ax.plot(range(len(signals)), signals)

        plt.show()