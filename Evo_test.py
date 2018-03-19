import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, EESN, ESN, DHESN
from MackeyGlass.MackeyGlassGenerator import run

from Helper.utils import nrmse
from EvolutionaryStrategies import RunES, RunGA

import datetime

if __name__ == '__main__':
    data = np.array([run(6100)]).reshape(-1, 1)
    MEAN_OF_DATA = np.mean(data)
    split = 5100
    X_train = data[:split-1]
    y_train = data[1:split]
    X_valid = data[split-1:-1]
    y_valid = data[split:]

    #=================================
    # NOTE: 1000 episodes is arbitrary.
    # I find that std < 0.05 works well and
    # that learning rate needs to be low, e.g. 0.001
    # For now it is pretty slow with large datasets.
    # Biggest problem is the random weight initialisations,
    # I get around this be making the number of duplicate
    # members in the population approx. 3 (see 'num_resample' param).
    #
    #=================================
    episodes = 1000
    name = "DHESN_GArun_3res"
    population = 10
    std = 0.01
    learn_rate = 0.001
    n = 3
    # base_esn = EESN(input_size=1, output_size=1, num_reservoirs=n, 
    #                 reservoir_sizes=np.linspace(10, 500, n, endpoint=True).astype(int),
    #                 regulariser=1e-4)
    base_esn = DHESN(1, 1, n,
                reservoir_sizes=np.linspace(200, 10, n, endpoint=True).astype(int), 
                echo_params=np.linspace(0.6, 0.1, n), 
                regulariser=1e-2, debug=True,
                # activation=(lambda x: x*(x>0).astype(float)),
                # activation=(lambda x: x),
                init_echo_timesteps=100, dims_reduce=(np.linspace(200, 50, n-1).astype(int).tolist()),
                # init_echo_timesteps=100, dims_reduce=(np.linspace(50, 200, n-1).astype(int).tolist()),
                encoder_type='VAE')
    base_esn.initialize_input_weights(scales=np.linspace(0.6, 0.2, n), strategies='uniform')
    base_esn.initialize_reservoir_weights(
                spectral_scales=np.linspace(1, 1.3, n),
                strategies=['uniform']*n,
                sparsity=0.1
                )
    # base_esn = ESN(input_size=1, output_size=1, reservoir_size=300, regulariser=1e-6)
    # base_esn.initialize_input_weights(scales=1.0)
    # base_esn.initialize_reservoir_weights(spectral_scales=np.linspace(1, 1.35, n, endpoint=True).tolist())
    # base_esn.train(X_train, y_train)
    y_pred = []

    # below is just a test to output the NRMSE of the BEST model so I can
    # compare before starting. Initial parameters for the model are set under
    # the 'Agent' class.
    # GENERATIVE =================================================
    # u_n_ESN = X_valid[0]
    # for _ in range(len(y_valid)):
    #     u_n_ESN = base_esn.forward(u_n_ESN)
    #     y_pred.append(u_n_ESN)

    # y_pred = np.array(y_pred).squeeze()
    # y_vals = y_valid.squeeze()
    # print(np.shape(y_pred))
    # print(np.shape(y_vals))
    # print(np.hstack((y_pred, y_vals)))
    # nrmse_err = nrmse(y_vals, y_pred, MEAN_OF_DATA)
    # print("NRMSE: {}".format(nrmse_err))


    # RunES(episodes, name, population, std, learn_rate,
    #         (X_train, y_train), (X_valid, y_valid), MEAN_OF_DATA, base_esn)

    # res_ranges = [(200, 10)]
    # echo_ranges = [(0.6, 0.1)]
    # weightin_ranges = [(0.6, 0.2)]
    # spect_ranges = [(1.0, 1.3)]

    params_base = np.zeros(3*n)
    params_base[:n] = np.linspace(0.6, 0.1, n, endpoint=True)
    params_base[n:(2*n)] = np.linspace(1., 1.3, n, endpoint=True)
    params_base[(2*n):(3*n)] = np.linspace(0.6, 0.2, n, endpoint=True)
    RunES(episodes, name, population, 0.01, 0.001,
            (X_train, y_train), (X_valid, y_valid), MEAN_OF_DATA, base_esn, 
            params_base=params_base)#,
            #verbose=True)
