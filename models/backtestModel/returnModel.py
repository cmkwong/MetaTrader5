from production.codes.models.backtestModel import indexModel
from production.codes.utils import tools
import pandas as pd
import numpy as np

# def get_earning_list(exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode):
#     start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))                   # discard the DateTimeIndex
#     earning = get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode).reset_index(drop=True)    # discard the DateTimeIndex
#     earnings = []
#     for s, e in zip(start_index, end_index):
#         earnings.append(np.sum(earning[s + 1: e + 1]))  # see notes point 6
#     return earnings

def get_ret_earning_list(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode, slsp=None):
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.DataFrame
    :param points_dff_values_df: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param signal: pd.Series
    :param long_mode: Boolean
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: rets (list), earnings (list)
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))               # discard the DateTimeIndex
    ret, earning = get_ret_earning(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, long_mode) # discard the DateTimeIndex
    ret.reset_index(drop=True)
    earning.reset_index(drop=True)
    rets, earnings = [], []
    for s, e in zip(start_index, end_index):
        ret_series, earning_series = ret[s + 1: e + 1], earning[s + 1: e + 1] # why added 1, see notes (6)
        if slsp != None:
            ret_series, earning_series = modify_ret_earning_with_SLSP(ret_series, earning_series, slsp[0], slsp[1]) # modify the return and earning if has stop-loss and stop-profit setting
        rets.append(ret_series.prod())
        earnings.append(np.sum(earning_series))
    return rets, earnings

def get_ret_earning(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, long_mode): # see note (45a)
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
    :param points_dff_values_df: points the change with respect to quote currency
    :param coefficient_vector: np.array
    :param long_mode: Boolean
    :return: pd.Series, pd.Series
    """
    modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)

    # ret
    change = (open_prices - open_prices.shift(1)) / open_prices.shift(1)
    olds = np.sum(np.abs(modified_coefficient_vector))
    news = (np.abs(modified_coefficient_vector) + (change * modified_coefficient_vector)).sum(axis=1)
    ret = pd.Series(news / olds, index=open_prices.index, name="return")

    # earning
    weighted_pt_diff = points_dff_values_df.values * modified_coefficient_vector.reshape(-1, )
    # calculate the price in required deposit dollar
    earning = pd.Series(np.sum(exchg_q2d.shift(1).values * weighted_pt_diff, axis=1), index=exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)

    return ret, earning

# def get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode):
#     """
#     :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
#     :param points_dff_values_df: points the change with respect to quote currency
#     :param coefficient_vector: np.array, raw vector with interception(constant value)
#     :param long_mode: Boolean
#     :return: pd.Series
#     """
#     modified_coefficient_vector = tools.get_modify_coefficient_vector(coefficient_vector, long_mode)
#     weighted_pt_diff = points_dff_values_df.values * modified_coefficient_vector.reshape(-1,)
#     # calculate the price in required deposit dollar
#     earning = pd.Series(np.sum(exchg_q2d.shift(1).values * weighted_pt_diff, axis=1), index=exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)
#     return earning

# def get_earning_by_signal(earning, signal):
#     """
#     :param earning: earning
#     :param signal: pd.Series (Boolean)
#     :return: pd.DataFrame
#     """
#     earning_by_signal = pd.Series(signal.shift(2).values * earning.values, index=signal.index, name="earning_by_signal").fillna(0.0) # shift 2 unit see (30e)
#     return earning_by_signal

def get_ret_earning_by_signal(ret, earning, signal, slsp=None):
    """
    :param ret: pd.Series
    :param earning: earning
    :param signal: pd.Series
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: pd.Series
    """
    ret_by_signal = pd.Series(signal.shift(2).values * ret.values, index=signal.index, name="ret_by_signal").fillna(1.0).replace({0: 1})
    earning_by_signal = pd.Series(signal.shift(2).values * earning.values, index=signal.index, name="earning_by_signal").fillna(0.0)  # shift 2 unit see (30e)
    if slsp != None:
        start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))
        for s, e in zip(start_index, end_index):
            s, e = s + 1, e + 1
            ret_by_signal.iloc[s:e], earning_by_signal.iloc[s:e] = modify_ret_earning_with_SLSP(ret.iloc[s:e], earning.iloc[s:e], slsp[0], slsp[1])
    return ret_by_signal, earning_by_signal

# def get_total_earning(earnings):
#     """
#     :param earnings: earning list
#     :return: float
#     """
#     total_earning = 0
#     for earning in earnings:
#         total_earning += earning
#     return total_earning

def get_total_ret_earning(rets, earnings):
    """
    :param rets: return list
    :param earnings: earning list
    :return: float, float
    """
    total_ret, total_earning = 1, 0
    for ret, earning in zip(rets, earnings):
        total_ret *= ret
        total_earning += earning
    return total_ret, total_earning

# def get_accum_earning(earning, signal):
#     """
#     :param signal: pd.Series
#     :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
#     :param points_dff_values_df: points the change with respect to quote currency
#     :param coefficient_vector: np.array, raw vector with interception(constant value)
#     :param long_mode: Boolean
#     :return: pd.Series with accumulative earning
#     """
#     earning_by_signal = get_earning_by_signal(earning, signal)
#     accum_earning = pd.Series(earning_by_signal.cumsum(), index=signal.index, name="accum_earning") # Simplify the function note 47a
#     return accum_earning

def get_accum_ret_earning(ret, earning, signal, slsp=None):
    """
    :param ret: pd.Series
    :param earning: pd.Series
    :param signal: pd.Series
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: accum_ret (pd.Series), accum_earning (pd.Series)
    """
    ret_by_signal, earning_by_signal = get_ret_earning_by_signal(ret, earning, signal, slsp)
    accum_ret = pd.Series(ret_by_signal.cumprod(), index=signal.index, name="accum_ret") # Simplify the function note 47a
    accum_earning = pd.Series(earning_by_signal.cumsum(), index=signal.index, name="accum_earning")  # Simplify the function note 47a
    return accum_ret, accum_earning

def modify_ret_earning_with_SLSP(ret_series, earning_series, sl, sp):
    """
    equation see 49b
    :param ret_series: pd.Series with numeric index
    :param earning_series: pd.Series with numeric index
    :param sl: stop-loss (negative value)
    :param sp: stop-profit (positive value)
    :return: ret (np.array), earning (np.array)
    """
    total = 0
    sl_buffer, sp_buffer = sl, sp
    ret_mask, earning_mask = np.ones((len(ret_series),)), np.zeros((len(ret_series),))
    for i, (r, e) in enumerate(zip(ret_series, earning_series)):
        total += e
        if total >= sp:
            ret_mask[i] = 1 + ((r-1)/e) * sp_buffer
            earning_mask[i] = sp_buffer
            break
        elif total <= sl:
            ret_mask[i] = 1 - ((1-r)/e) * sl_buffer
            earning_mask[i] = sl_buffer
            break
        else:
            ret_mask[i], earning_mask[i] = ret_series[i], earning_series[i]
            sl_buffer -= e
            sp_buffer -= e
    return ret_mask, earning_mask

# def get_ret_earning_with_SLSP(signal, exchg_q2d, open_price, points_dff_values_df, coefficient_vector, long_mode, sl, sp):
#     """
#     :param signal: pd.Series
#     :param exchg_q2d: pd.DataFrame
#     :param open_price: pd.DataFrame
#     :param points_dff_values_df: pd.DataFrame
#     :param coefficient_vector: np.array
#     :param long_mode: Boolean
#     :param sl: stop-loss (negative value)
#     :param sp: stop-profit (positive value)
#     :return: ret_by_signal(pd.Series) and earning_by_signal(pd.Series), are modified with stop-loss and stop-profit
#     """
#     ret = get_ret(open_price, coefficient_vector, long_mode)
#     earning = get_earning(exchg_q2d, points_dff_values_df, coefficient_vector, long_mode)
#     ret_by_signal = get_ret_by_signal(ret, signal)
#     earning_by_signal = get_earning_by_signal(earning, signal)
#
#     # get the start index and end index
#     start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))
#     for s, e in zip(start_index, end_index):
#         s, e = s+1, e+1
#         ret_by_signal.iloc[s:e], earning_by_signal.iloc[s:e] = modify_ret_earning_with_SLSP(ret.iloc[s:e], earning.iloc[s:e], sl, sp)
#     return ret_by_signal.rename("ret_by_signal_slsp"), earning_by_signal.rename("earning_by_signal_slsp")




