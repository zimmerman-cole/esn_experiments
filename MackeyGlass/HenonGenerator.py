import matplotlib.pyplot as plt
import numpy as np

from collections import deque # for deque

import datetime

# henon params
alpha = 1.4
beta = 0.3

def runHenon(num_data_samples=5000, dimensions=1):

    assert dimensions == 1 or dimensions == 2, "HENON MAP CAN ONLY be 1D or 2D"

        # move data
    #x_t = 0.1
    #y_t = 0.3
    x_t = 0.0
    y_t = 0.0

        # record timesteps
    sample_timer = 0

        # sample for training
    #num_data_samples = 5000
    current_sample = 0

    data_samples = np.zeros((num_data_samples, dimensions))

    init_period = 2000

    while True:

        if dimensions == 1:
            x_t_next = x_t_plus_one_1d(x_t, y_t)
            y_t = x_t
            x_t = x_t_next
        elif dimensions == 2:
            x_t_next = x_t_plus_one(x_t, y_t)
            y_t_next = y_t_plus_one(x_t, y_t)
            x_t = x_t_next
            y_t = y_t_next

        # store data
        if sample_timer > init_period:
            data_samples[current_sample, 0] = x_t
            if dimensions == 2:
                data_samples[current_sample, 1] = y_t
            current_sample += 1
            if current_sample >= num_data_samples:
                print("DONE HENON")
                return data_samples

        sample_timer += 1

def onExit(data):
    # save the data
    np.savetxt("data_{}.txt".format(datetime.date.today()), data, delimiter=",")

    # plot the data
    if np.shape(data)[1] == 2:
        plt.scatter(data[:, 1], data[:, 0], s=0.4)
        plt.xlabel("x_t")
        plt.ylabel("y_t")
        plt.show()
    else:
        plt.plot(range(len(data)), data)
        plt.show()

def x_t_plus_one(x_t, y_t):
    return 1. - alpha * x_t*x_t + y_t

def y_t_plus_one(x_t, y_t):
    return beta * x_t

def x_t_plus_one_1d(x_t, x_t_minus_one):
    return 1. - alpha * x_t*x_t + beta * x_t_minus_one

if __name__ == "__main__":
    data = runHenon(num_data_samples=2000, dimensions=1)
    onExit(data)
