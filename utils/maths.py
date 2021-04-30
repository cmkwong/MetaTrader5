import numpy as np

def z_col(col):
    mean = np.mean(col)
    std = np.std(col)
    normalized_col = (col - mean) / std
    return normalized_col