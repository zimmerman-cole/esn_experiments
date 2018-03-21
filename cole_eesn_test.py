import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import EESN, ESN
from MackeyGlass.MackeyGlassGenerator import run
from Helper.utils import nrmse
import pickle as pkl
import itertools
import time

def mse(y1, y2):
    return np.mean((y1 - y2)**2)
                        
if __name__ == '__main__':
    data = np.array([run(15100)]).reshape(-1, 1)
    data_mean = np.mean(data, axis=0)
    split = 14100
    X_train = np.array(data[:split-1])
    y_train = np.array(data[1:split])
    X_valid = np.array(data[split-1:-1])
    y_valid = np.array(data[split:])
    data_mean = np.mean(data)

    #esn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    #esn.initialize_input_weights(scale=1.0)
    #esn.initialize_reservoir_weights(spectral_scale=1.25)

    num_reservoirs = 10
    echo_params_ = [
        np.linspace(0.85, 0.5, num_reservoirs), np.linspace(0.85, 0.85, num_reservoirs), 
        np.linspace(0.9, 0.9, num_reservoirs), np.linspace(0.9, 0.5, num_reservoirs),
        np.linspace(0.9, 0.75, num_reservoirs), np.linspace(0.75, 0.9, num_reservoirs),
        np.linspace(0.5, 0.9, num_reservoirs), np.linspace(0.5, 0.85, num_reservoirs)
    ]
    regulariser_ = [1e-2, 1e-3, 1e-4]
    spectral_scales_ = [
        np.linspace(0.9, 0.7, num_reservoirs), np.linspace(1.0, 1.0, num_reservoirs),
        np.linspace(0.9, 1.25, num_reservoirs), np.linspace(0.5, 1.25, num_reservoirs)
    ]
    input_scales_ = [
        np.linspace(0.2, 0.2, num_reservoirs), np.linspace(0.5, 0.5, num_reservoirs),
        np.linspace(0.2, 1.0, num_reservoirs), np.linspace(1.0, 0.2, num_reservoirs)
    ]
    reservoir_sizes_ = [
        np.linspace(10, 500, num_reservoirs).astype(int),
        np.linspace(200, 200, num_reservoirs).astype(int),
        np.linspace(100, 200, num_reservoirs).astype(int),
        np.linspace(50, 250, num_reservoirs).astype(int)
    ]
    iterables = [
        echo_params_, regulariser_, spectral_scales_, input_scales_, reservoir_sizes_
    ]

    title = time.asctime().replace(' ', '-')
    to_save = dict()
    for settings in itertools.product(*iterables):
        echo_params, regulariser, spectral_scales, input_scales, reservoir_sizes = settings
        print('='*40)
        print('echo_prms', echo_params)
        print('reg', regulariser)
        print('spec_scales', spectral_scales)
        print('in_scales', input_scales)
        print('res_sizes', reservoir_sizes)

        for _ in range(3):
            eesn = EESN(
                1, 1, num_reservoirs, reservoir_sizes=reservoir_sizes, echo_params=echo_params, 
                regulariser=regulariser, debug=True
            )
            eesn.initialize_input_weights(scales=input_scales)
            eesn.initialize_reservoir_weights(spectral_scales=spectral_scales)

            #esn.train(X_train, y_train)
            eesn.train(X_train, y_train)

            #esn_outputs = []
            eesn_outputs = []

            # GENERATIVE =================================================
            #u_n_ESN = data[split]
            u_n_EESN = np.array(X_valid[0])
            for _ in range(len(data[split:])):
                #u_n_ESN = esn.forward(u_n_ESN)
                #esn_outputs.append(u_n_ESN)
                u_n_EESN = eesn.forward(u_n_EESN)
                eesn_outputs.append(u_n_EESN)

            #esn_outputs = np.array(esn_outputs).squeeze()
            eesn_outputs = np.array(eesn_outputs).squeeze()

            error = nrmse(y_valid, eesn_outputs, data_mean)
            #print('  ESN MSE: %f' % mse(y_valid, esn_outputs))
            print('EESN NRMSE: %f' % error)

            key = eesn.info()
            if key not in to_save.keys():
                to_save[key] = []
            to_save[key].append(error)
            
            if 0:
                f, ax = plt.subplots(figsize=(12, 12))
                #ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
                ax.plot(range(len(eesn_outputs)), eesn_outputs, label='EESN')
                ax.plot(range(len(y_valid)), y_valid, label='True')
                plt.legend()
                plt.show()

                #for res in eesn.reservoirs:
                #    all_signals = np.array(res.signals).squeeze()[:100, :5].T

                #    f, ax = plt.subplots()
                #    ax.set_title(u'Reservoir %d. \u03B5: %.2f' % (res.idx, res.echo_param))
                #    for signals in all_signals:
                #        ax.plot(range(len(signals)), signals)

                #    plt.show()

            if 0:
                f, ax = plt.subplots(figsize=(12, 12))
                w = eesn.W_out.squeeze()
                ax.bar(range(len(w)), w)
                plt.show()

            pkl.dump(to_save, open('Results/EESN/%s.p' % title, 'wb'))
