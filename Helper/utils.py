import numpy as np
import matplotlib.pyplot as plt
import time
import os

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def nrmse(y_true, y_pred, MEAN_OF_DATA):
    return np.sqrt(np.sum((y_true - y_pred)**2)/np.sum((y_true - MEAN_OF_DATA)**2))


# Default ESN specifications
_DEFAULT_SPECS_ = {
    'echo_params': 0.85,
    'regulariser': 1e-5,
    'num_reservoirs': 5,
    'reservoir_sizes': 200,
    'in_weights': {'strategies': 'binary', 'scales': 0.2, 'offsets': 0.5},
    'res_weights': {'strategies': 'uniform', 'spectral_scales': 1., 'offsets': 0.5}
}
    