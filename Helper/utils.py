import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import os

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def nrmse(y_true, y_pred, MEAN_OF_DATA):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return np.sqrt(np.sum((y_true - y_pred)**2)/np.sum((y_true - MEAN_OF_DATA)**2))

class LiveDataGraph:

    def __init__(self, data_function, update_rate=200, title=""):
        self.data_function = data_function
        x_data, y_data = self.data_function()
        self.figure = plt.figure(figsize=(4,4))#4*len(self.data_function)))
        self.figure.suptitle(title)
        self.line, = plt.plot(x_data, y_data)
        self.update_rate = update_rate
        self.animation = FuncAnimation(self.figure, self.__update__, interval=self.update_rate)

        # start the graphing process in a new thread
        # t = threading.Thread(target=self.__run__)
        # t.start()
        # self.__run__()

    def __update__(self, frame):
        x_data, y_data = self.data_function()
        self.line.set_data(x_data, y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()
        return self.line

    def run(self):
        plt.show(block=False)
        plt.pause(0.001)

class LivePlotHistogram:

    def __init__(self, data_functions, names, update_rate=1, barCount=50, title=""):
        self.data_functions = data_functions
        self.names = names
        self.figure, self.axs = plt.subplots(len(self.data_functions), 1, figsize=(4,4))
        self.figure.suptitle(title)
        self.title = title
        self.update_rate = update_rate
        self.barCount = barCount
        self.centres = []
        self.widths = []
        self.hists = []
        
        self.timestep = 0

    def __update__(self):
        # if self.timestep > 0:
            # self.figure.clf()
            # # self.figure.suptitle(self.title)
            # self.figure.gca().autoscale_view()

        self.centres = []
        self.widths = []
        self.hists = []
        for i in self.data_functions:
            y_data = i()
            # print(y_data)
            h, bars = np.histogram(y_data, bins=self.barCount)
            c = (bars[1:] + bars[:-1]) / 2.0
            w = (bars[0] - bars[1]) * 0.8
            self.centres.append(c)
            self.widths.append(w)
            self.hists.append(h)

    def run(self):
        if self.timestep % self.update_rate == 0:
            self.__update__()
            # idx = 0
            for idx, (h,c,w) in enumerate(zip(self.hists, self.centres, self.widths)):
                # print(idx)
                self.axs[idx].clear()
                self.axs[idx].bar(c, h, width=w)
                self.axs[idx].set_title(self.names[idx])
                # self.axs[idx].plot(range(0, 100), range(0, 100))

            plt.show(block=False)
            plt.pause(0.001)

        self.timestep += 1
        

# Default ESN specifications
_DEFAULT_SPECS_ = {
    'echo_params': 0.85,
    'regulariser': 1e-5,
    'num_reservoirs': 5,
    'reservoir_sizes': 200,
    'in_weights': {'strategies': 'binary', 'scales': 0.2, 'offsets': 0.5},
    'res_weights': {'strategies': 'uniform', 'spectral_scales': 1., 'offsets': 0.5}
}
    
