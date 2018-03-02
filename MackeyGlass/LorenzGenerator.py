import matplotlib.pyplot as plt
import numpy as np

from collections import deque # for deque

import datetime

# Global variables
    # speed
move_speed = 0.008

    # mackey glass params
sigma = 10.0
phi = 28.0
beta = 8./3.

def runLorenz(num_data_samples=5000):
        # move data
    # x_pos = 0.2
    # y_pos = 0.2
    # z_pos = 0.2

    #     # record timesteps
    # sample_timer = 0

    #     # sample for training
    # #num_data_samples = 5000
    # current_sample = 0
    # data_samples = np.zeros((num_data_samples, 3))

    # init_period = 300

    # while True:

    #     x_next = x_pos + move_speed * d_x(x_pos, y_pos, z_pos)
    #     y_next = y_pos + move_speed * d_y(x_pos, y_pos, z_pos)
    #     z_next = z_pos + move_speed * d_z(x_pos, y_pos, z_pos)
    #     x_pos = x_next
    #     y_pos = y_next
    #     z_pos = z_next

    #     # store data
    #     if sample_timer > init_period:
    #         data_samples[current_sample, 0] = x_pos
    #         data_samples[current_sample, 1] = y_pos
    #         data_samples[current_sample, 2] = z_pos
    #         current_sample += 1
    #         if current_sample >= num_data_samples:
    #             print("DONE")
    #             return data_samples

    #     sample_timer += 1

    #------ better version on wikipedia which just solves the ODE----
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    from mpl_toolkits.mplot3d import Axes3D

    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

    delta_t = 0.01
    t_end = num_data_samples * delta_t
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, t_end, delta_t)

    states = odeint(f, state0, t)

    return states

def onExit(data, axis1=0, axis2=2):
    # save the data
    np.savetxt("data_{}.txt".format(datetime.date.today()), data, delimiter=",")

    # plot the data
    plt.plot(data[:, axis1], data[:, axis2], linewidth=0.4)
    plt.xlabel("{}".format(axis1))
    plt.ylabel("{}".format(axis2))
    plt.show()

def d_x(x_t, y_t, z_t):
    return sigma * (y_t - x_t)

def d_y(x_t, y_t, z_t):
    return x_t * (phi - z_t) - y_t

def d_z(x_t, y_t, z_t):
    return x_t * y_t - beta * z_t

if __name__ == "__main__":
    data = runLorenz(num_data_samples=20000)
    onExit(data)
