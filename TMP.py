import time
import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import ESN2
from ESN.ESN import ESN, LCESN, EESN
from MackeyGlass.MackeyGlassGenerator import run, onExit

def run_TESTS():
    data = np.array(run(21100)).reshape(-1, 1)
    data_mean = np.mean(data)
    split = 20000

    X_train = data[:split-1]
    y_train = data[1:split]
    X_test = data[split-1:-1]
    y_test = data[split:]

    lcesn = LCESN(input_size=1, output_size=1, num_reservoirs=3, reservoir_sizes=300, echo_params=0.85, 
                init_echo_timesteps=100, regulariser=1e-6)
    lcesn.initialize_input_weights(strategies='binary', scales=1.)
    lcesn.initialize_reservoir_weights(strategies='uniform', spectral_scales=1.25)
    print('LCESN MADE')
    eesn = EESN(input_size=1, output_size=1, num_reservoirs=3, reservoir_sizes=300, echo_params=0.85, 
                init_echo_timesteps=100, regulariser=1e-6)
    eesn.initialize_input_weights(strategies='binary', scales=1.)
    eesn.initialize_reservoir_weights(strategies='uniform', spectral_scales=1.25)
    print('EESN MADE')
    print('='*30)

    st_time = time.time()
    lcesn.train(X_train, y_train)
    print('LCESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

    st_time = time.time()
    eesn.train(X_train, y_train)
    print('EESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

    lcesn_outs = []
    eesn_outs = []
    for i, x in enumerate(X_test):
        lcesn_outs.append(lcesn.forward(x))
        eesn_outs.append(eesn.forward(x))

    lcesn_outs = np.array(lcesn_outs).squeeze()
    eesn_outs = np.array(eesn_outs).squeeze()

    fig, ax = plt.subplots()
    ax.plot(range(len(lcesn_outs)), lcesn_outs, label='lcesn')
    ax.plot(range(len(eesn_outs)), eesn_outs, label='new')
    ax.plot(range(len(y_test)), y_test, label='true')
    plt.show()



def run_ESN():
    data = np.array(run(21100)).reshape(-1, 1)
    data_mean = np.mean(data)
    split = 20000

    X_train = data[:split-1]
    y_train = data[1:split]
    X_test = data[split-1:-1]
    y_test = data[split:]

    OLD_ESN = ESN2(input_size=1, output_size=1, reservoir_size=1000, echo_param=0.85, spectral_scale=1.25,
                init_echo_timesteps=100, regulariser=1e-6, input_weights_scale=1.)
    print('OLD ESN MADE')

    NEW_ESN = ESN(input_size=1, output_size=1, reservoir_size=1000, echo_param=0.85, 
                init_echo_timesteps=100, regulariser=1e-6)
    NEW_ESN.initialize_input_weights(strategy='binary', scale=1.)
    NEW_ESN.initialize_reservoir_weights(strategy='uniform', spectral_scale=1.25)
    print('NEW ESN MADE')
    print('='*30)

    # Set new ESN's weights = old ESN's weights to ensure they (SHOULD) output the same outputs
    NEW_ESN.reservoir.W_res = OLD_ESN.W_reservoir
    NEW_ESN.reservoir.W_in = OLD_ESN.W_in

    assert np.sum(abs(NEW_ESN.reservoir.W_res - OLD_ESN.W_reservoir)) < 1e-3

    if 0:
        for x0 in X_train:
            x0 = x0.reshape(-1, 1)
            old_res_fwd, in_old, res_old, old_st_old = OLD_ESN.__forward_to_res__(x0, debug=True)
            new_res_fwd, in_new, res_new, old_st_new = NEW_ESN.reservoir.forward(x0, debug=True)
            print('init state diff', np.sum(old_res_fwd - new_res_fwd))
            print(' in_to_res diff: ', np.sum(in_old - in_new))
            print('res_to_res diff: ', np.sum(res_old - res_new))
            print(' old state diff: ', np.sum(old_st_old - old_st_new))
            raw_input()

    # TRAIN THE NETWORKS =================================================
    st_time = time.time()
    OLD_ESN.train(X_train, y_train) # S, D
    print('OLD ESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

    st_time = time.time()
    NEW_ESN.train(X_train, y_train)
    print('NEW ESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

    #print('success?', (OLD_ESN.W_out == NEW_ESN.W_out))

    # diffs = (NEW_ESN.W_out - OLD_ESN.W_out).flatten()

    #plt.plot(range(len(diffs)), diffs)
    #plt.show()

    old_outputs = []
    new_outputs = []
    for i, x in enumerate(X_test):
        old_outputs.append(OLD_ESN.forward_to_out(x))
        new_outputs.append(NEW_ESN.forward(x))

    old_outputs = np.array(old_outputs).squeeze()
    new_outputs = np.array(new_outputs).squeeze()

    #print(old_outputs.shape)
    #print(old_outputs.shape, old_outputs[0].shape, old_outputs[0])

    # if 1:
    #     f, axarr = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 12))
    #     axarr[0].plot(range(len(old_outputs)), old_outputs, label='old')
    #     axarr[1].plot(range(len(new_outputs)), new_outputs, label='new')
    #     axarr[2].plot(range(len(y_test)), y_test, label='true')
    #     plt.legend(); plt.show()
    #     f.close()
    fig, ax = plt.subplots()
    ax.plot(range(len(old_outputs)), old_outputs, label='old')
    ax.plot(range(len(new_outputs)), new_outputs, label='new')
    ax.plot(range(len(y_test)), y_test, label='true')
    plt.show()


if __name__ == "__main__":
    run_TESTS()

