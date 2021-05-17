from production.codes.models.backtestModel import indexModel
import pandas as pd
import numpy as np

# def get_ret_list2(open_price, signal):
#     """
#     :param open_price: pd.Series
#     :param signal: pd.Series(Boolean)
#     :return: float
#     """
#     start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
#     ret = get_ret2(open_price).reset_index(drop=True) # discard the DateTimeIndex
#     rets = []
#     for s, e in zip(start_index, end_index):
#         rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
#     return rets
#
# def get_ret2(open_price):
#     """
#     :return: open_price: pd.Series
#     """
#     diffs = open_price.diff(periods=1)
#     shifts = open_price.shift(1)
#     ret = 1 + diffs / shifts
#     return ret

# def get_rets_df_debug(open_prices):
#     rets = pd.DataFrame(index=open_prices.index)
#     for name in open_prices.columns:
#         rets[name] = get_ret(open_prices, exchg_q2d, modified_coefficient_vector, long_mode)
#     return rets

def get_earning_list(exchg_q2d, points_dff_values_df, modified_coefficient_vector, signal):
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))  # discard the DateTimeIndex
    earning = get_earning(exchg_q2d, points_dff_values_df, modified_coefficient_vector).reset_index(drop=True)  # discard the DateTimeIndex
    earnings = []
    for s, e in zip(start_index, end_index):
        earnings.append(np.sum(earning[s + 1: e + 1]))  # see notes point 6
    return earnings

def get_ret_list(open_prices, exchg_q2d, modified_coefficient_vector, signal, long_mode):
    """
    :param open_prices: pd.Series
    :param signal: pd.Series(Boolean)
    :return: float
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
    ret = get_ret(open_prices, exchg_q2d, modified_coefficient_vector, long_mode).reset_index(drop=True) # discard the DateTimeIndex
    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
    return rets

def get_ret(open_prices, exchg_q2d, modified_coefficient_vector, long_mode):
    """
    :param open_prices:
    :param exchg_q2d:
    :param coefficient_vector:
    :return:
    """
    old_value_df = (open_prices * modified_coefficient_vector.reshape(-1,) * exchg_q2d.values).sum(axis=1).shift(1)
    new_value_df = (open_prices * modified_coefficient_vector.reshape(-1,) * exchg_q2d.shift(1).values).sum(axis=1)
    if long_mode:
        ret = pd.Series(new_value_df / old_value_df, index=open_prices.index, name='long')
    else:
        ret = pd.Series(old_value_df / new_value_df, index=open_prices.index, name='short')
    return ret

# def get_weighted_ret_df2(open_prices, exchg_q2d, modified_coefficient_vector):
#     """
#     Calculate the return in same deposit
#     :param open_prices: pd.Dataframe, open prices
#     :param exchg_q2d: exchange rate from base to deposit
#     :param modified_coefficient_vector: np.array, coefficient (if no weighted, pass the array([1]) if long, pass the array([-1]) if short)
#     :return: weighted_value_df
#     """
#     long_spread_weight_factor = np.append(-1 * modified_coefficient_vector[1:], 1)   # buy real, sell predict
#     short_spread_weight_factor = np.append(modified_coefficient_vector[1:], -1)      # buy predict, sell real
#     old_value_df, new_value_df, ret_df = pd.DataFrame(index=open_prices.index), pd.DataFrame(index=open_prices.index), pd.DataFrame(index=open_prices.index)
#     old_value_df['long'] = (open_prices * long_spread_weight_factor * exchg_q2d.values).sum(axis=1).shift(1)
#     old_value_df['short'] = (open_prices * short_spread_weight_factor * exchg_q2d.values).sum(axis=1).shift(1)
#     new_value_df['long'] = (open_prices * long_spread_weight_factor * exchg_q2d.shift(1).values).sum(axis=1)
#     new_value_df['short'] = (open_prices * short_spread_weight_factor * exchg_q2d.shift(1).values).sum(axis=1)
#     ret_df['long'] = new_value_df['long'] / old_value_df['long']
#     ret_df['short'] = old_value_df['short'] / new_value_df['short']
#     return ret_df

def get_earning(exchg_q2d, points_dff_values_df, modified_coefficient_vector, deposit_symbol='USD'):
    """
    :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
    :param points_dff_values_df: points the change with respect to quote currency
    :param modified_coefficient_vector: two type of coefficient vector, short and long
    :return:
    """
    spread_weighted_pt_diff = points_dff_values_df.values * modified_coefficient_vector.reshape(-1,)
    # calculate the price in required deposit dollar
    earning = pd.Series(np.sum(exchg_q2d.shift(1).values * spread_weighted_pt_diff, axis=1), index=exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)
    return earning

def get_earning_by_signal(earning, signal):
    """
    :param earning: earning
    :param signal: pd.Series (Boolean)
    :return: pd.DataFrame
    """
    # earning_by_signal = pd.DataFrame(index=signal.index)
    # for name in signal.columns:
        # signal.loc[:, name] = signalModel.discard_head_signal(signal[name])
        # signal.loc[:, name] = signalModel.discard_tail_signal(signal[name])
    earning_by_signal = pd.Series(signal.shift(2).values * earning.values, index=signal.index, name="earning_by_signal") # shift 2 unit see (30e)
    return earning_by_signal

def get_accum_earning(earnings):
    accum_earning = 0
    for earning in earnings:
        accum_earning += earning
    return accum_earning

def get_accum_ret(rets):
    accum_ret = 1
    for ret in rets:
        accum_ret *= ret
    return accum_ret

# def get_accum_earning2(exchg_q2d, points_dff_values_df, modified_coefficient_vector, signal):
#     earning = get_earning(exchg_q2d, points_dff_values_df, modified_coefficient_vector)
#     earning_by_signal = get_earning_by_signal(earning, signal)
#     accum_earning = earning_by_signal.sum(axis=0)
#     return accum_earning
#
# def get_accum_ret2(open_price, signal):
#     """
#     :param open_price: pd.Series
#     :param signal: pd.Series(Boolean)
#     :return: accum_ret: float64
#     """
#     accum_ret = 1
#     rets = get_ret_list(open_price, signal)
#     for ret in rets:
#         accum_ret *= ret
#     return accum_ret