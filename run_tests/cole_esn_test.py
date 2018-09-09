import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run
from Helper.utils import nrmse
import pickle as pkl
import itertools
import time
 
if __name__ == '__main__':
    data = np.array([run(15100)]).reshape(-1, 1)
    data_mean = np.mean(data, axis=0)
    split = 14100
    X_train = np.array(data[:split-1])
    y_train = np.array(data[1:split])
    X_valid = np.array(data[split-1:-1])
    y_valid = np.array(data[split:])
    data_mean = np.mean(data)

    echo_params_ = np.linspace(0.2, 0.9, 7).tolist()
    regulariser_ = [1e-2, 1e-4, 1e-6, 1e-8]
    spectral_scales_ = np.linspace(0.2, 1.25, 10).tolist()
    input_scales_ = np.linspace(0.2, 1.0, 10).tolist()
    reservoir_sizes_ = [500, 750, 1000, 1250]
    sparsities_ = [0.1, 0.5, 1.0]
    iterables = [
        echo_params_, regulariser_, spectral_scales_, input_scales_, reservoir_sizes_, sparsities_
    ]

    title = time.asctime().replace(' ', '-')
    to_save = dict()
    for settings in itertools.product(*iterables):
        echo_params, regulariser, spectral_scales, input_scales, reservoir_sizes, sp = settings
        print('='*40)
        print('echo_prm', echo_params)
        print('reg', regulariser)
        print('spec_scale', spectral_scales)
        print('in_scale', input_scales)
        print('res_size', reservoir_sizes)
        print('sparsity', sp)

        for _ in range(3):
            esn = ESN(1, 1, reservoir_sizes, echo_params, regulariser=regulariser)
            esn.initialize_input_weights(scale=input_scales)
            esn.initialize_reservoir_weights(spectral_scale=spectral_scales, sparsity=sp)

            esn.train(X_train, y_train)

            esn_outputs = []

            # GENERATIVE =================================================
            u_n_ESN = np.array(X_valid[0])
            for _ in range(len(y_valid)):
                u_n_ESN = np.array(esn.forward(u_n_ESN))
                esn_outputs.append(u_n_ESN)

            esn_outputs = np.array(esn_outputs).squeeze()

            error = nrmse(y_valid, esn_outputs, data_mean)
            print('ESN NRMSE: %f' % error)

            key = esn.info()
            if key not in to_save.keys():
                to_save[key] = []
            to_save[key].append(error)
            
            if 0:
                f, ax = plt.subplots(figsize=(12, 12))
                #ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
                ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
                ax.plot(range(len(y_valid)), y_valid, label='True')
                plt.legend()
                plt.show()

                #for res in esn.reservoirs:
                #    all_signals = np.array(res.signals).squeeze()[:100, :5].T

                #    f, ax = plt.subplots()
                #    ax.set_title(u'Reservoir %d. \u03B5: %.2f' % (res.idx, res.echo_param))
                #    for signals in all_signals:
                #        ax.plot(range(len(signals)), signals)

                #    plt.show()

            if 0:
                f, ax = plt.subplots(figsize=(12, 12))
                w = esn.W_out.squeeze()
                ax.bar(range(len(w)), w)
                plt.show()

            pkl.dump(to_save, open('Results/ESN/%s.p' % title, 'wb'))
