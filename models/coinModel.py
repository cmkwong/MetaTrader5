import numpy as np
import pandas as pd
from production.codes.utils import maths, tools
from production.codes.models import priceModel
from production.codes.models.backtestModel import signalModel, returnModel

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

def get_strategy_id(train_options):
    id = 'coin'
    for key, value in train_options.items():
        id += str(value)
    long_id = id + 'long'
    short_id = id + 'short'
    return long_id, short_id

def get_action(trader, strategy_id, masked_open_prices, quote_exchg, ptDv, coefficient_vector, signal, slsp, long_mode):
    """
    :param trader: Class Trader object
    :param strategy_id: str, each strategy has unique id for identity
    :param masked_open_prices: open price with last price masked by current price
    :param quote_exchg: pd.DataFrame
    :param ptDv: exchg: pd.DataFrame
    :param coefficient_vector: np.array
    :param signal: pd.Series
    :param slsp: tuple, (stop-loss, stop-profit)
    :param long_mode: Boolean
    :return: None
    """
    # long spread
    if signal[-2] == True and signal[-3] == False and trader.status[strategy_id] == 0:
        trader.strategy_execute(strategy_id, pos_open=True)      # open position
    elif signal[-2] == False and signal[-3] == True and trader.status[strategy_id] == 1:
        trader.strategy_execute(strategy_id, pos_open=False)     # close position
    elif trader.status[strategy_id] == 1:
        _, earning = returnModel.get_ret_earning(masked_open_prices, quote_exchg, ptDv, coefficient_vector, long_mode=long_mode)
        _, accum_earning = returnModel.get_accum_ret_earning(_, earning, signal)
        if accum_earning[-1] > slsp[1]:
            trader.strategy_execute(strategy_id, pos_open=False)   # close position
        elif accum_earning[-1] < slsp[0]:
            trader.strategy_execute(strategy_id, pos_open=False)
    # # short spread
    # if short_signal[-2] == 1 and short_signal[-3] == 0 and short_status == 0:
    #     return short_lots # open short-spread
    # elif short_signal[-2] == 0 and short_signal[-3] == 1 and short_status == 1:
    #     return long_lots # close short-spread
    # elif short_status == 1:
    #     _, short_earning = returnModel.get_ret_earning(masked_open_prices, quote_exchg, ptDv, coefficient_vector, long_mode=False)
    #     _, short_accum_earning = returnModel.get_accum_ret_earning(_, short_earning, signal)
    #     if short_accum_earning < slsp[0]:
    #         return long_lots # close short-spread

