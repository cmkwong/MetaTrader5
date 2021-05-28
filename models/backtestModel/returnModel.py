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

def get_ret_list(open_prices, coefficient_vector, signal, long_mode):
    """
    :param open_prices: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param signal: pd.Series
    :param long_mode: Boolean
    :return: return list
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
    ret = get_ret(open_prices, coefficient_vector, long_mode).reset_index(drop=True)     # discard the DateTimeIndex
    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # why added 1, see notes (6)
    return rets

def get_ret(open_prices, coefficient_vector, long_mode): # see note (45a)
    """
    :param open_prices: pd.DataFrame
    :param coefficient_vector: np.array
    :param long_mode: Boolean
    :return: pd.Series
    """
    modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)
    change = (open_prices - open_prices.shift(1)) / open_prices.shift(1)
    olds = np.sum(np.abs(modified_coefficient_vector))
    news = (np.abs(modified_coefficient_vector) + (change * modified_coefficient_vector)).sum(axis=1)
    ret = pd.Series(news / olds, index=open_prices.index, name="return")
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

def get_accum_earning(signal, exchg_q2d, points_dff_values_df, coefficient_vector, long_mode):
    """
    :param signal: pd.Series
    :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
    :param points_dff_values_df: points the change with respect to quote currency
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param long_mode: Boolean
    :return: pd.Series with accumulative earning
    """
    earning = get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode)
    earning_by_signal = get_earning_by_signal(earning, signal)
    accum_earning = pd.Series(earning_by_signal.cumsum(), index=signal.index, name="accum_earning") # Simplify the function note 47a
    return accum_earning

def get_accum_ret(signal, open_price, coefficient_vector, long_mode):
    """
    :param signal: pd.Series
    :param open_prices: pd.DataFrame
    :param coefficient_vector: np.array
    :param long_mode: Boolean
    :return: pd.Series with accumulative return
    """
    ret = get_ret(open_price, coefficient_vector, long_mode)
    ret_by_signal = get_ret_by_signal(ret, signal)
    accum_ret = pd.Series(ret_by_signal.cumprod(), index=signal.index, name="accum_ret") # Simplify the function note 47a
    return accum_ret
