import numpy as np


def median_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps

    return np.median(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))
