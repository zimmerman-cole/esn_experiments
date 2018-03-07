import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def nrmse(y_true, y_pred, MEAN_OF_DATA):
    return np.sqrt(np.sum((y_true - y_pred)**2)/np.sum((y_true - MEAN_OF_DATA)**2))