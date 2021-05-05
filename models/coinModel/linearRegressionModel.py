import numpy as np

def get_x_vector(input, target):
    """
    :param input: array, size = (total_len, )
    :param target: array, size = (total_len, )
    :return: x
    """
    A = np.concatenate((np.ones((len(input), 1), dtype=float), input.reshape(-1, 1)), axis=1)
    b = target.reshape(-1, 1)
    A_T_A = np.dot(np.transpose(A), A)
    A_T_b = np.dot(np.transpose(A), b)
    x = np.dot(np.linalg.inv(A_T_A), A_T_b)
    return x

def get_predicted_arr(input, x):
    """
    :param input: array, size = (total_len, )
    :param x: array, size = (feature_size, )
    :return: predicted array
    """
    A = np.concatenate((np.ones((len(input), 1)), input.reshape(-1,1)), axis=1)
    b = np.dot(A, x.reshape(-1,1))
    return b