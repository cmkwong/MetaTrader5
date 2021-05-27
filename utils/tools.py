import numpy as np

def shift_list(lst, s):
    s %= len(lst)
    s *= -1
    shifted_lst = lst[s:] + lst[:s]
    return shifted_lst

def split_matrix(arr, percentage=0.8, axis=0):
    """
    :param arr: np.array() 2D
    :param percentage: float
    :param axis: float
    :return: split array
    """
    cutOff = int(arr.shape[axis] * percentage)
    max = arr.shape[axis]
    I = [slice(None)] * arr.ndim
    I[axis] = slice(0, cutOff)
    upper_arr = arr[tuple(I)]
    I[axis] = slice(cutOff, max)
    lower_arr = arr[tuple(I)]
    return upper_arr, lower_arr

def split_df(df, percentage):
    split_index = int(len(df) * percentage)
    upper_df = df.iloc[:split_index,:]
    lower_df = df.iloc[split_index:, :]
    return upper_df, lower_df

def get_modify_coefficient_vector(coefficient_vector, long_mode):
    """
    :param coefficient_vector: np.array, if empty array, it has no coefficient vector -> 1 or -1
    :param long_mode: Boolean
    :return: np.array
    """
    if long_mode:
        modified_coefficient_vector = np.append(-1 * coefficient_vector[1:], 1)  # buy real, sell predict
    else:
        modified_coefficient_vector = np.append(coefficient_vector[1:], -1)  # buy predict, sell real
    return modified_coefficient_vector.reshape(-1,)

def get_accuracy(values, th=0.0):
    """
    :param values: list
    :param th: float
    :return: float
    """
    accuracy = np.sum([c > th for c in values]) / len(values)
    return accuracy