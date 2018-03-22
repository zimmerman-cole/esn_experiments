import matplotlib.pyplot as plt
import numpy as np

from collections import deque # for deque

import datetime

# Global variables
    # speed
move_speed = 1.0 

    # mackey glass params
gamma = 0.1
beta = 0.2
tau = 17 
en = 10.

print("hello")
#def resetHistory():
    #x_history.clear()
    #x_history = deque(maxlen=tau)

#def addToHistory(x):
    #x_history.append(x)

def run(num_data_samples=5000, init_x=1.0, init_x_tau=0.0):
    x_history = deque(maxlen=tau)
    x_history.clear()
    #resetHistory()
        # move data
    x_pos = init_x
    x_pos_tau = init_x_tau

        # record timesteps
    sample_timer = 0

        # sample for training
    #num_data_samples = 5000
    current_sample = 0
    data_samples = []

    while True:
        if len(x_history) >= tau:
            x_pos_tau = x_history[0]

        x_pos += move_speed * d_x(x_pos, x_pos_tau)

        # store data
        if sample_timer > 300:
            data_samples.append(x_pos)
            current_sample += 1
            if current_sample >= num_data_samples:
                #print("DONE")
                return data_samples

        # record move history
        x_history.append(x_pos)

        sample_timer += 1

def onExit(data, plot2d=False):
    # save the data
    data_np = np.asarray(data)
    np.savetxt("data_{}.txt".format(datetime.date.today()), data_np, delimiter=",")

    # plot the data
    if plot2d:
        x_minus_tau = data[:-tau]
        x_ = data[tau:]
        plt.plot(x_, x_minus_tau, linewidth=0.2)
        plt.xlabel("x(t)")
        plt.ylabel("x(t-tau)")
        plt.show()
    else:
        plt.plot(range(len(data)), data)
        plt.show()

def d_x(x_t, x_t_tau):
    return beta * (x_t_tau/(1.+pow(x_t_tau, en))) - gamma * x_t

if __name__ == "__main__":
    data = run(num_data_samples=8000)
    onExit(data, plot2d=True)
