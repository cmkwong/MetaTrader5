from production.codes.models.backtestModel import indexModel
import pandas as pd
import numpy as np

def get_ret_list(open_price, signal):
    """
    :param open_price: pd.Series
    :param signal: pd.Series(Boolean)
    :return: float
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
    ret = get_ret(open_price).reset_index(drop=True) # discard the DateTimeIndex
    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
    return rets

def get_ret(open_price):
    """
    :return: open_price: pd.Series
    """
    diffs = open_price.diff(periods=1)
    shifts = open_price.shift(1)
    ret = 1 + diffs / shifts
    return ret

def get_rets_df_debug(open_prices):
    rets = pd.DataFrame(index=open_prices.index)
    for name in open_prices.columns:
        rets[name] = get_ret(open_prices[name])
    return rets

def get_weighted_ret_list(open_prices, exchg_q2d, coefficient_vector, signal):
    """
    :param open_prices: pd.Series
    :param signal: pd.Series(Boolean)
    :return: float
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))   # discard the DateTimeIndex
    ret = get_weighted_ret_df(open_prices, exchg_q2d, coefficient_vector).reset_index(drop=True) # discard the DateTimeIndex

    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
    return rets

def get_weighted_ret_df(open_prices, exchg_q2d, coefficient_vector):
    """
    Calculate the return in same deposit
    :param open_prices: pd.Dataframe, open prices
    :param exchg_q2d: exchange rate from base to deposit
    :param coefficient_vector: np.array, coefficient (if no weighted, pass the array([1]) if long, pass the array([-1]) if short)
    :return: weighted_value_df
    """
    long_spread_weight_factor = np.append(-1 * coefficient_vector[1:], 1)   # buy real, sell predict
    short_spread_weight_factor = np.append(coefficient_vector[1:], -1)      # buy predict, sell real
    old_value_df, new_value_df, ret_df = pd.DataFrame(index=open_prices.index), pd.DataFrame(index=open_prices.index), pd.DataFrame(index=open_prices.index)
    old_value_df['long'] = np.sum(open_prices.values * long_spread_weight_factor * exchg_q2d.values, axis=1)
    old_value_df['short'] = np.sum(open_prices.values * short_spread_weight_factor * exchg_q2d.values, axis=1)
    new_value_df['long'] = np.sum(open_prices.values * long_spread_weight_factor * exchg_q2d.shift(1).values, axis=1)
    new_value_df['short'] = np.sum(open_prices.values * short_spread_weight_factor * exchg_q2d.shift(1).values, axis=1)
    ret_df['long'] = new_value_df['long'] / old_value_df['long'].shift(1)
    ret_df['short'] = old_value_df['short'].shift(1) / new_value_df['short']
    return ret_df

def get_accum_ret(open_price, signal):
    """
    :param open_price: pd.Series
    :param signal: pd.Series(Boolean)
    :return: accum_ret: float64
    """
    accum_ret = 1
    rets = get_ret_list(open_price, signal)
    for ret in rets:
        accum_ret *= ret
    return accum_ret

def get_accum_earning(earning, signal):
    """
    :param earning: pd.Series, earning changed from open price
    :param signal: Series(Boolean)
    :earningurn: earning_by_signal: float64
    """
    earning_by_signal = signal.shift(2) * earning
    accum_earning = earning_by_signal.sum(axis=0)
    return accum_earning
