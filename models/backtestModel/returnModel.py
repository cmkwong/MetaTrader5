from production.codes.models.backtestModel import indexModel, pointsModel
from production.codes.models import coinModel
import pandas as pd
import numpy as np

def get_ret_earning_list(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode, slsp=None, lot_times=1):
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.DataFrame
    :param points_dff_values_df: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param signal: pd.Series
    :param long_mode: Boolean
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :param lot_times: lot lot_times
    :return: rets (list), earnings (list)
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))               # discard the DateTimeIndex
    ret, earning = get_ret_earning(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, long_mode, lot_times) # discard the DateTimeIndex
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

def get_ret_earning(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, long_mode, lot_times=1): # see note (45a)
    """
    :param open_prices: pd.DataFrame
    :param exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
    :param points_dff_values_df: points the change with respect to quote currency
    :param coefficient_vector: np.array
    :param long_mode: Boolean
    :param lot_times: lot times
    :return: pd.Series, pd.Series
    """
    modified_coefficient_vector = coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode, lot_times)

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

def get_value_of_ret_earning(symbols, new_values, old_values, q2d_at, coefficient_vector, all_symbols_info, long_mode, lot_times):
    """
    This is calculate the return and earning from raw value (instead of from dataframe)
    :param symbols: [str]
    :param new_values: np.array (Not dataframe)
    :param old_values: np.array (Not dataframe)
    :param q2d_at: np.array, values at brought the assert
    :param coefficient_vector: np.array
    :param all_symbols_info: nametuple
    :param long_mode: Boolean
    :return: float, float: ret, earning
    """

    modified_coefficient_vector = coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode, lot_times)

    # ret value
    changes = (new_values - old_values) / old_values
    olds = np.sum(np.abs(modified_coefficient_vector))
    news = (np.abs(modified_coefficient_vector) + (changes * modified_coefficient_vector)).sum()
    ret = news / olds

    # earning value
    points_dff_values = pointsModel.get_points_dff_from_values(symbols, new_values, old_values, all_symbols_info)
    weighted_pt_diff = points_dff_values * modified_coefficient_vector.reshape(-1, )
    # calculate the price in required deposit dollar
    earning = np.sum(q2d_at * weighted_pt_diff)

    # prices_at
    prices_at = old_values

    return ret, earning, prices_at

# def get_ret_earning_priceAt_after_close_position(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode, lot_times):
#     """
#     :param open_prices: pd.DataFrame
#     :param exchg_q2d: pd.DataFrame
#     :param points_dff_values_df: pd.DataFrame
#     :param coefficient_vector: np.array
#     :param signal: pd.Series
#     :param long_mode: Boolean
#     :param lot_times: int
#     :return: ret, earning, prices_at (float, float, np.array)
#     """
#     ret_list, earning_list = get_ret_earning_list(open_prices, exchg_q2d, points_dff_values_df, coefficient_vector, signal, long_mode, lot_times=lot_times)
#     ret, earning = ret_list[-1], earning_list[-1]  # extract the last value in the series
#     prices_at = list(open_prices.iloc[-1, :])
#     return ret, earning, prices_at