import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, EESN, ESN, DHESN
from MackeyGlass.HenonGenerator import runHenon
from MackeyGlass.MackeyGlassGenerator import run

from Helper.utils import nrmse

import datetime

# def mse(y1, y2):
#     return np.mean((y1-y2)**2)

# def nrmse(y_true, y_pred, MEAN_OF_DATA):
#     return np.sqrt(np.sum((y_true - y_pred)**2)/np.sum((y_true - MEAN_OF_DATA)**2))

def save_data(file_name, data_csv, delimiter, fmt, header):
    np.savetxt(file_name, data_csv, delimiter=delimiter, 
                fmt=fmt, header=header)


if __name__ == '__main__':
    data = np.  array([run(15100)]).reshape(-1, 1)
    # NOTE: REMOVE WHEN NOT DHESN
    _std = np.std(data)
    #data -= np.mean(data)
    # data /= _std
    MEAN_OF_DATA = np.mean(data)
    split = 14100
    X_train = np.array(data[:split-1])
    y_train = np.array(data[1:split])
    X_valid = np.array(data[split-1:-1])
    y_valid = np.array(data[split:])

    # print(np.mean(X_train))
    # print(np.mean(X_valid))

    # eesn = ESN(1, 1, 1000, echo_param=0.76800719, regulariser=1e-5)
    # eesn.initialize_input_weights(scale=1.00567055)
    # eesn.initialize_reservoir_weights(spectral_scale=1.19223381)
    # eesn = ESN(1, 1, 1000, echo_param=0.85, regulariser=1e-6)
    # eesn.initialize_input_weights(scale=1.0)
    # eesn.initialize_reservoir_weights(spectral_scale=1.25)

    # eesn.train(X_train, y_train)

    # res_ranges = [(10, 500), (110, 600)]
    # echo_ranges = [(0.1, 0.8), (0.8, 0.1), (0.8, 0.8), (0.3, 0.5), (0.5, 0.9)]
    # spect_ranges = [(1.0, 1.5), (1.5, 1.0), (1.0, 1.25)]


    # eesn = ESN(1, 1, 5, reservoir_sizes=300, echo_params=[0.65388205, 0.28477042, 0.22879262, 0.12106287, 0.8],
    #             regulariser=1e-5, debug=True, activation=(lambda x: x))
    # eesn.initialize_input_weights(scales=[1.16899269, 0.73521864, 0.28610932, 0.36676156, 1.88768737])
    # eesn.initialize_reservoir_weights(spectral_scales=[1.55865201, 1.0, 0.51742484, 0.58150598, 1.0])
    # eesn.train(X_train, y_train)
    # eesn = EESN(1, 1, 5, reservoir_sizes=300, echo_params=0.8,
    #             regulariser=1e-5, debug=True, activation=(lambda x: x))
    # eesn.initialize_input_weights(scales=0.8)
    # eesn.initialize_reservoir_weights(spectral_scales=1.0)
    # eesn.train(X_train, y_train)

    # eesn_outputs = []

    # GENERATIVE =================================================
    # u_n_EESN = data[split-1]
    # for _ in range(len(data[split:])):
    #     u_n_EESN = eesn.forward(u_n_EESN)
    #     eesn_outputs.append(u_n_EESN)

    # eesn_outputs = np.array(eesn_outputs).squeeze()
    # y_vals = y_valid.squeeze()
    # nrmse_err = nrmse(y_vals, eesn_outputs, MEAN_OF_DATA)
    # print('EESN NRMSE: %f' % nrmse_err)

    EXPERIMENT_NAME = "DHESN_WITH_VAE_GRID_SEARCH_epochfix_3_nostd_"
    file_name = 'DHESN_RESULTS/DHESN_data_{}_{}.csv'.format(EXPERIMENT_NAME, datetime.date.today())

    res_ranges = [(100, 500), (200, 400), (300, 300)]
    echo_ranges = [(0.8, 0.5), (0.5, 0.1)]
    weightin_ranges = [(1.0, 0.5), (0.5, 0.5)]
    spect_ranges = [(0.4, 1.2), (1.2, 0.4), (1.0, 0.7)]
    d_reduce_ranges = [(60, 60), (10, 100), (30, 80)]
    res_number_ranges = [4, 8]
    regs = [1e-2, 1e-6]
    train_epochs = [2, 5]
    # num_res = 20
    # reg = 1e-6

    num_samples = 3

    data_samples = np.zeros((len(res_ranges)*len(echo_ranges)*len(spect_ranges)*len(res_number_ranges)*len(weightin_ranges)*len(d_reduce_ranges)*len(regs)*len(train_epochs), 1+2+2+2+2+2+1+1+1+1))
    data_csv = np.zeros((len(res_ranges)*len(echo_ranges)*len(spect_ranges)*len(res_number_ranges)*len(weightin_ranges)*len(d_reduce_ranges)*len(regs)*len(train_epochs), 1+2+2+2+2+2+1+1+1+1))
    total_runs = len(res_ranges)*len(echo_ranges)*len(spect_ranges)*len(res_number_ranges)*len(weightin_ranges)*len(d_reduce_ranges)*len(regs)*len(train_epochs)
    runs = 200 
    nrmses_d = []
    idx = 0
    for n in res_number_ranges:
        for r in res_ranges:
            for w in weightin_ranges:
                for e in echo_ranges:
                    for s in spect_ranges:
                        for d in d_reduce_ranges:
                            for reg in regs:
                                for t in train_epochs:
                                    _errors = []
                                    _reservoirs = np.round(np.linspace(r[0], r[1], n, endpoint=True).astype(int), 3)
                                    _echoes = np.round(np.linspace(e[0], e[1], n, endpoint=True), 3)
                                    _spectrals = np.round(np.linspace(s[0], s[1], n, endpoint=True), 3).tolist()
                                    _weightins = np.round(np.linspace(w[0], w[1], n, endpoint=True), 3).tolist()
                                    _dimsreduce = np.linspace(d[0], d[1], n-1).astype(int).tolist()
                                    # _echoes = np.array([0.2618, 0.6311, 0.2868, 0.6311, 0.2868, 0.6311, 0.2868, 0.6311])
                                    # _spectrals = np.array([0.8896, 0.8948, 0.3782, 0.8948, 0.3782, 0.8948, 0.3782, 0.8948])
                                    # _weightins = np.array([0.7726, 0.4788, 0.6535, 0.4788, 0.6535, 0.4788, 0.6535, 0.4788])
                                    # _echoes = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
                                    # _weightins = [0.2, 1.0, 1.0, 1.0, 1.0]
                                    # print(range(10, 100, n-1)[::-1])
                                    print("({}/{}) EXPERIMENT: \n\tRES: {}, \n\tECH: {}, \n\tSPEC: {}, \n\tWEIGHTIN: {}, \n\tDIMREDUC: {}, \n\tREG: {}, \n\tEPOCH: {}".format(
                                                    idx, total_runs, _reservoirs, _echoes, _spectrals, _weightins, _dimsreduce, reg, t))
                                    for l in range(num_samples):
                                        # r = (np.sqrt((1000**2)/n), np.sqrt((1000**2)/n))
                                        # _reservoirs = np.round(np.linspace(r[0], r[1], n, endpoint=True).astype(int), 3)
                                        # _echoes = np.round(np.linspace(e[0], e[1], n, endpoint=True), 3)
                                        # _spectrals = np.round(np.linspace(s[0], s[1], n, endpoint=True), 3).tolist()
                                        # _weightins = np.round(np.linspace(w[0], w[1], n, endpoint=True), 3).tolist()
                                        # _dimsreduce = np.linspace(d[0], d[1], n-1).astype(int).tolist()
                                        # # _echoes = np.array([0.2618, 0.6311, 0.2868, 0.6311, 0.2868, 0.6311, 0.2868, 0.6311])
                                        # # _spectrals = np.array([0.8896, 0.8948, 0.3782, 0.8948, 0.3782, 0.8948, 0.3782, 0.8948])
                                        # # _weightins = np.array([0.7726, 0.4788, 0.6535, 0.4788, 0.6535, 0.4788, 0.6535, 0.4788])
                                        # # _echoes = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
                                        # # _weightins = [0.2, 1.0, 1.0, 1.0, 1.0]
                                        # # print(range(10, 100, n-1)[::-1])
                                        # print("EXPERIMENT: \n\tRES: {}, \n\tECH: {}, \n\tSPEC: {}, \n\tWEIGHTIN: {}, \n\tDIMREDUC: {}, \n\tREG: {}".format(
                                        #                 _reservoirs, _echoes, _spectrals, _weightins, _dimsreduce, reg))
                                        eesn = DHESN(1, 1, n,
                                                    reservoir_sizes=_reservoirs, 
                                                    echo_params=_echoes, 
                                                    regulariser=reg, debug=True,
                                                    # activation=(lambda x: x*(x>0).astype(float)),
                                                    # activation=(lambda x: x),
                                                    init_echo_timesteps=100, dims_reduce=_dimsreduce,
                                                    # init_echo_timesteps=100, dims_reduce=(np.linspace(50, 200, n-1).astype(int).tolist()),
                                                    encoder_type='VAE', train_epochs=t, train_batches=64)
                                        eesn.initialize_input_weights(scales=_weightins, strategies='uniform')
                                        eesn.initialize_reservoir_weights(
                                                    spectral_scales=_spectrals,
                                                    strategies=['uniform']*n,
                                                    sparsity=0.1
                                                    )
                                        # eesn = ESN(1, 1, reservoir_size=1000,
                                        #              echo_param=0.2,
                                        #              regulariser=1e-5, debug=False,
                                        #              # activation=(lambda x: x*(x>0).astype(float)),
                                        #              # activation=(lambda x: x),
                                        #              init_echo_timesteps=100)
                                        #              # init_echo_timesteps=100, dims_reduce=(np.linspace(50, 200, n-1).astype(int).tolist()),
                                        # eesn.initialize_input_weights(scale=.5)
                                         #eesn.reservoir.W_in[:, -1] += MEAN_OF_DATA
                                        # eesn.initialize_reservoir_weights(
                                        #              spectral_scale=5.5,
                                        #              sparsity=.1)
                                        eesn.train(X_train, y_train)

                                        eesn_outputs = []

                                        # GENERATIVE =================================================
                                        # u_n_EESN = data[split-1]
                                        u_n_EESN = X_valid[0]
                                        for _ in range(len(data[split:])):
                                            u_n_EESN = eesn.forward(u_n_EESN)
                                            eesn_outputs.append(u_n_EESN)

                                        # SUPERVISED ====
                                        # for u_n in X_valid:
                                        #     esn_outputs.append(esn.forward(u_n))
                                        #     lcesn_outputs.append(lcesn.forward(u_n))
                                        # ============================================================

                                        # esn_outputs = np.array(esn_outputs).squeeze()
                                        eesn_outputs = np.array(eesn_outputs).squeeze()
                                        y_vals = y_valid.squeeze()
                                        # print(np.shape(eesn_outputs))
                                        # print(np.shape(y_vals))
                                        # print(np.vstack((eesn_outputs, y_vals)))

                                        # print('  ESN MSE: %f' % mse(y_valid, esn_outputs))
                                        # print('EESN MSE: %f' % mse(y_valid, eesn_outputs))
                                        nrmse_err = nrmse(y_vals, eesn_outputs, MEAN_OF_DATA)
                                        if nrmse_err > 100000:
                                            nrmse_err = 100000
                                        print('DHESN NRMSE: %f' % nrmse_err)
                                        #print('-log(NRMSE): %f' % -np.log(nrmse_err))
                                        _errors.append(nrmse_err)
                                        # plt.plot(range(len(eesn_outputs)), eesn_outputs, label="predicted")
                                        # plt.plot(range(len(y_vals)), y_vals, label="true")
                                        # plt.legend()
                                        # plt.show()

                                        # we = eesn.W_out.squeeze()
                                        # plt.bar(range(len(we)), we)
                                        # plt.show()

                                    data_csv[idx, 0] = n
                                    data_csv[idx, 1:3] = r
                                    data_csv[idx, 3:5] = e
                                    data_csv[idx, 5:7] = s
                                    data_csv[idx, 7:9] = w
                                    data_csv[idx, 9:11] = d
                                    data_csv[idx, 11] = t
                                    data_csv[idx, 12] = reg
                                    data_csv[idx, 13] = np.mean(_errors)
                                    data_csv[idx, 14] = np.std(_errors)
                                    idx += 1

                                    # save data periodically
                                    if idx % 20 == 0:
                                        save_data(file_name, data_csv, ',',
                                                    ['%d', '%d', '%d', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%d', '%.2e', '%.4f', '%.6e'],
                                                    'No. res, res sizes (min), (max), echo values (min), (max), '+
                                                    'spectral values (min), (max), weightin (min), (max), dimsrec (min), (max), epochs, reg., NRMSE (mean), (std)')
                                        print("!!!DATA SAVED!!!")
                                        
    
    # font = {'family' : 'normal',
    #     # 'weight' : 'bold',
    #     'size'   : 22}
    # plt.rc('font', **font)
    # plt.rc('legend', fontsize=10)

    # colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'black', 'brown']
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    # for i, e in enumerate(eesn.encoders):
    #     s = e.sample()
    #     a = e.avg_loss_history
    #     # print("sample {}: {}".format(i, s))
    #     ax1.plot(range(len(s)), s - i*3, color=colors[i], label='VAE {}'.format(i))
    #     ax2.plot(range(len(a)), a, color=colors[i], label='VAE {}'.format(i))
    # ax1.legend()
    # ax1.set_xlabel("sampled unit Guassian")
    # ax1.set_ylabel("latent variables")
    # ax2.legend()
    # ax2.set_xlabel("epoch")
    # ax2.set_ylabel("avg. MSE loss")

    # fig3, ax3 = plt.subplots()
    # hist, bins = np.histogram(eesn.W_out, bins=50)
    # centres = (bins[1:]+bins[:-1])/2.
    # width = (bins[1] - bins[0])*0.9
    # print(np.shape(eesn.W_out))
    # ax3.bar(range(len(eesn.W_out.squeeze())), eesn.W_out.squeeze(), width=0.8)

    # plt.show()

    # plt.plot(range(len(X_train)), X_train, label="XTRAIN")
    # plt.plot(range(len(y_train)), y_train, label="yTRAIN")
    # plt.plot(range(len(X_valid)), X_valid, label="XVALID")
    # plt.plot(range(len(y_valid)), y_valid, label="yVALID")
    # plt.show()

    # save the data
    #file_name = 'DHESN_RESULTS/DHESN_data_{}_{}.csv'.format(EXPERIMENT_NAME, datetime.date.today())
    #np.savetxt(file_name, data_csv, delimiter=',', 
    #           fmt=['%d', '%d', '%d', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.2e', '%.4f', '%.6e'],
    #           header='No. res, res sizes (min), (max), echo values (min), (max), '+
    #           'spectral values (min), (max), weightin (min), (max), dimsrec (min), (max), reg., NRMSE (mean), (std)')
    save_data(file_name, data_csv, ',',
                ['%d', '%d', '%d', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%d', '%.2e', '%.4f', '%.6e'],
                'No. res, res sizes (min), (max), echo values (min), (max), '+
                'spectral values (min), (max), weightin (min), (max), dimsrec (min), (max), epochs, reg., NRMSE (mean), (std)')

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

