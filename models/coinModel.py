import numpy as np
import pandas as pd
from production.codes.utils import maths

def get_coefficient_vector(input, target):
    """
    :param input: array, size = (total_len, )
    :param target: array, size = (total_len, )
    :return: coefficient
    """
    A = np.concatenate((np.ones((len(input), 1), dtype=float), input.reshape(len(input), -1)), axis=1)
    b = target.reshape(-1, 1)
    A_T_A = np.dot(np.transpose(A), A)
    A_T_b = np.dot(np.transpose(A), b)
    coefficient_vector = np.dot(np.linalg.inv(A_T_A), A_T_b)
    return coefficient_vector

def get_predicted_arr(input, coefficient_vector):
    """
    Ax=b
    :param input: array, size = (total_len, feature_size)
    :param coefficient_vector: coefficient vector, size = (feature_size, )
    :return: predicted array
    """
    A = np.concatenate((np.ones((len(input), 1)), input.reshape(len(input),-1)), axis=1)
    b = np.dot(A, coefficient_vector.reshape(-1,1)).reshape(-1,)
    return b

def get_coin_data(close_prices, coefficient_vector, mean_window, std_window):
    """
    :param close_prices: accept the train and test prices in pd.dataframe format
    :param coefficient_vector:
    :return:
    """
    coin_data = pd.DataFrame(index=close_prices.index)
    coin_data['real'] = close_prices.iloc[:,-1]
    coin_data['predict'] = get_predicted_arr(close_prices.iloc[:,:-1].values, coefficient_vector)
    spread = coin_data['real'] - coin_data['predict']
    coin_data['spread'] = spread
    coin_data['z_score'] = maths.z_score_with_rolling_mean(spread.values, mean_window, std_window)
    return coin_data

def check_action(close_price_with_last_tick_l, close_price_with_last_tick_s, coefficient_vector, z_score_mean_window, z_score_std_window, upper_th, lower_th, slsp):
    """
    :param close_price_with_last_tick_l: pd.DataFrame for long that replace with last unit of price into tick price
    :param close_price_with_last_tick_s: pd.DataFrame for short that replace with last unit of price into tick price
    :param coefficient_vector: np.array
    :param z_score_mean_window: int
    :param z_score_std_window: int
    :param upper_th: float
    :param lower_th: float
    :param slsp: tuple, stop loss and stop profit
    :return: long_spread, short_spread, close all deals
    """
    long_coin_data = get_coin_data(close_price_with_last_tick_l, coefficient_vector, z_score_mean_window, z_score_std_window)
    if long_coin_data['z_score'][-1] < lower_th:
        return 'long_spread'
    short_coin_data = get_coin_data(close_price_with_last_tick_s, coefficient_vector, z_score_mean_window, z_score_std_window)
    if short_coin_data['z_score'][-1] > upper_th:
        return 'short_spread'

