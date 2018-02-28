import time
import numpy as np
import matplotlib.pyplot as plt
from ESN.ESN import ESN2
from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run, onExit

data = np.array(run(21100)).reshape(-1, 1)
data_mean = np.mean(data)

X = data[:-1]
y = data[1:]

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

for x0 in X:
    x0 = x0.reshape(-1, 1)
    old_res_fwd, in_old, res_old = OLD_ESN.__forward_to_res__(x0)
    new_res_fwd, in_new, res_new = NEW_ESN.reservoir.forward(x0)
    print('init state diff', np.sum(old_res_fwd - new_res_fwd))
    print(' in_to_res diff: ', np.sum(in_old - in_new))
    print('res_to_res diff: ', np.sum(res_old - res_new))
    raw_input()

# TRAIN THE NETWORKS =================================================
st_time = time.time()
X_train, y_target = OLD_ESN.train(X, y) # S, D
print('OLD ESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

st_time = time.time()
S, D = NEW_ESN.train(X, y)
print('NEW ESN TRAINED. TOOK %.3f SEC' % (time.time() - st_time))

x = np.array([[1.]])
old_out = OLD_ESN.forward_to_out(x, debug=True)
new_out = NEW_ESN.forward(x, debug=True)
print(old_out, new_out)

print('success?', (OLD_ESN.W_out == NEW_ESN.W_out))

diffs = (NEW_ESN.W_out - OLD_ESN.W_out).flatten()

#plt.plot(range(len(diffs)), diffs)
#plt.show()

print('S DIFF: ')
print(np.sum(S - X_train))
print('D DIFF: ')
print(np.sum(D - y_target))

