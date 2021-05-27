from production.codes.models.backtestModel import indexModel
from production.codes.utils import tools
import pandas as pd
import numpy as np

def get_earning_list(exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode):
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))                   # discard the DateTimeIndex
    earning = get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode).reset_index(drop=True)    # discard the DateTimeIndex
    earnings = []
    for s, e in zip(start_index, end_index):
        earnings.append(np.sum(earning[s + 1: e + 1]))  # see notes point 6
    return earnings

def get_ret_list(open_prices, exchg_q2d, coefficient_vector, signal, long_mode):
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param signal: pd.Series
    :param long_mode: Boolean
    :return: return list
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
    ret = get_ret(open_prices, exchg_q2d, coefficient_vector, long_mode).reset_index(drop=True)     # discard the DateTimeIndex
    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
    return rets

def get_ret(open_prices, _, coefficient_vector, long_mode): # see note 45a

    modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)
    change = (open_prices - open_prices.shift(1)) / open_prices.shift(1)
    olds = np.sum(np.abs(modified_coefficient_vector))
    news = (np.abs(modified_coefficient_vector) + (change * modified_coefficient_vector)).sum(axis=1)
    ret = news / olds
    return ret

def get_ret2(open_prices, exchg_q2d, coefficient_vector, long_mode):
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param long_mode: Boolean
    :return: pd.Series
    """
    modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)
    old_value_df = (open_prices * modified_coefficient_vector.reshape(-1,) * exchg_q2d.values).sum(axis=1).shift(1)
    new_value_df = (open_prices * modified_coefficient_vector.reshape(-1,) * exchg_q2d.shift(1).values).sum(axis=1) # use past exchange rate as reference rate (Note 34a&b)
    ret = pd.Series(1 + ((new_value_df - old_value_df) / old_value_df.abs()), index=open_prices.index, name='return') # absolute the value (note 44a and side note)
    return ret

def get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode):
    """
    :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
    :param points_dff_values_df: points the change with respect to quote currency
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param long_mode: Boolean
    :return: pd.Series
    """
    modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)
    weighted_pt_diff = points_dff_values_df.values * modified_coefficient_vector.reshape(-1,)
    # calculate the price in required deposit dollar
    earning = pd.Series(np.sum(exchg_q2d.shift(1).values * weighted_pt_diff, axis=1), index=exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)
    return earning

def get_earning_by_signal(earning, signal):
    """
    :param earning: earning
    :param signal: pd.Series (Boolean)
    :return: pd.DataFrame
    """
    earning_by_signal = pd.Series(signal.shift(2).values * earning.values, index=signal.index, name="earning_by_signal").fillna(0.0) # shift 2 unit see (30e)
    return earning_by_signal

def get_ret_by_signal(ret, signal):
    """
    :param ret: pd.Series
    :param signal: pd.Series
    :return: pd.Series
    """
    ret_by_signal = pd.Series(signal.shift(2).values * ret.values, index=signal.index, name="ret_by_signal").fillna(1.0).replace({0: 1})
    return ret_by_signal

def get_total_earning(earnings):
    """
    :param earnings: earning list
    :return: float
    """
    total_earning = 0
    for earning in earnings:
        total_earning += earning
    return total_earning

def get_total_ret(rets):
    """
    :param rets: return list
    :return: float
    """
    total_ret = 1
    for ret in rets:
        total_ret *= ret
    return total_ret

def get_accum_earning(earning, signal):
    """
    :param earning: pd.Series
    :param signal: pd.Series
    :return: pd.Series
    """
    earning_by_signal = get_earning_by_signal(earning, signal)
    # rolling
    accum_value = 0
    accum_earning_list = []
    for value in earning_by_signal:
        accum_value += value
        accum_earning_list.append(accum_value)
    accum_earning = pd.Series(accum_earning_list, index=signal.index, name="accum_earning")
    return accum_earning

def get_accum_ret(ret, signal):
    """
    :param ret: pd.Series
    :param signal: pd.Series
    :return: pd.Series
    """
    ret_by_signal = get_ret_by_signal(ret, signal)
    # rolling
    accum_value = 1
    accum_ret_list = []
    for value in ret_by_signal:
        accum_value *= value
        accum_ret_list.append(accum_value)
    accum_ret = pd.Series(accum_ret_list, index=signal.index, name="accum_ret")
    return accum_ret
