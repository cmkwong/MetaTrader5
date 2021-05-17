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

def get_coin_data(close_prices, coefficient_vector, windows=3):
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
    coin_data['z_score'] = maths.z_score_with_rolling_mean(spread.values, windows)
    return coin_data


# def get_current_spread(coefficient_vector, symbols, timeframe, timezone, start):
#     """
#     :param coefficient_vector: coefficient vector, size = (feature_size, )
#     :param symbols: [str]
#     :param timeframe: mt5.timeframe
#     :param timezone: str: "Hongkong"
#     :param count: int
#     :return: spread
#     """
#     price_matrix = mt5Model.get_prices_df(symbols, timeframe, timezone, start).values
#     b = get_predicted_arr(price_matrix[:,:-1], coefficient_vector)
#     spread = price_matrix[:,-1].reshape(-1,) - b
#     return spread

# data_options = {
#     'start': (2010,1,1,0,0),
#     'end': (2021,5,4,0,0),
#     'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "USDCAD"],
#     'timeframe': mt5Model.get_txt2timeframe('H4'),
#     'timezone': "Hongkong",
#     'shuffle': True,
#     'trainTestSplit': 0.7,
# }

# coefficient_vector = np.array([2.3894, 0.01484, -1.338143, -0.015469])
# spread = get_current_spread(coefficient_vector, data_options['symbols'], data_options['timeframe'], data_options['timezone'], data_options['start'])
# reslut = maths.perform_ADF_test(spread)
# z_scores = maths.z_score_with_rolling_mean(spread, 10)
# print()