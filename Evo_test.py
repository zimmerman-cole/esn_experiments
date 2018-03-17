import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import LCESN, EESN, ESN
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
    name = "EESN_GArun1"
    population = 15
    std = 0.01
    learn_rate = 0.001
    n = 10
    base_esn = EESN(input_size=1, output_size=1, num_reservoirs=n, 
                    reservoir_sizes=np.linspace(10, 500, n, endpoint=True).astype(int),
                    regulariser=1e-4)
    # base_esn = ESN(input_size=1, output_size=1, reservoir_size=300, regulariser=1e-6)
    base_esn.initialize_input_weights(scales=1.0)
    base_esn.initialize_reservoir_weights(spectral_scales=np.linspace(1, 1.35, n, endpoint=True).tolist())
    base_esn.train(X_train, y_train)
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

    params_base = np.zeros(3*n)
    params_base[:n] = 0.8
    params_base[n:(2*n)] = np.linspace(1, 1.35, n, endpoint=True)
    params_base[(2*n):(3*n)] = 1.0
    RunGA(episodes, name, population,
            (X_train, y_train), (X_valid, y_valid), MEAN_OF_DATA, base_esn, 
            params_base=params_base,
            verbose=True)
