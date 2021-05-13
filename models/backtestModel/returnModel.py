from production.codes.models.backtestModel import indexModel
import pandas as pd

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

# def get_ret(df):
#     """
#     :return: ret: Series
#     """
#     ret = get_ret(df)
#     ret = 1 + ret
#     return ret

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
