import numpy as np
import pandas as pd
from production.codes.models import mt5Model

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
    :param long_mode: Boolean, True = long spread, False = short spread
    :return: np.array
    """
    if long_mode:
        modified_coefficient_vector = np.append(-1 * coefficient_vector[1:], 1)  # buy real, sell predict
    else:
        modified_coefficient_vector = np.append(coefficient_vector[1:], -1)  # buy predict, sell real
    return modified_coefficient_vector.reshape(-1,)

def get_close_price_with_last_tick(close_price, coefficient_vector):
    """
    :param close_price: pd.DataFrame
    :param coefficient_vector: np.array
    :return: dict with pd.DataFrame
    """
    long_modified_coefficient_vector = get_modify_coefficient_vector(coefficient_vector, long_mode=True)

    # re-create the dataframe
    close_price_with_last_tick = {}
    close_price_with_last_tick['long_spread'] = close_price.copy() # why using copy(), see note 55b
    close_price_with_last_tick['short_spread'] = close_price.copy()
    symbols = close_price.columns
    for i, symbol in enumerate(symbols):
        lasttick = mt5Model.get_last_tick(symbol)
        if long_modified_coefficient_vector[i] >= 0:
            close_price_with_last_tick['long_spread'].iloc[-1, i] = lasttick['ask']
            close_price_with_last_tick['short_spread'].iloc[-1, i] = lasttick['bid']
        else:
            close_price_with_last_tick['long_spread'].iloc[-1, i] = lasttick['bid']
            close_price_with_last_tick['short_spread'].iloc[-1, i] = lasttick['ask']
    return close_price_with_last_tick

def get_accuracy(values, th=0.0):
    """
    :param values: listclose_price_with_last_tick
    :param th: float
    :return: float
    """
    accuracy = np.sum([c > th for c in values]) / len(values)
    return accuracy