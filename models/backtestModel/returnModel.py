from production.codes.models.backtestModel import indexModel


def get_ret_list(df, signal):
    """
    :param signal: Series(Boolean)
    :return: float
    """
    start_index, end_index = indexModel.get_action_start_end_index(signal)
    ret = get_ret(df)
    rets = []
    for s, e in zip(start_index, end_index):
        rets.append(ret[s + 1: e + 1].prod())  # see notes point 6
    return rets

def get_change(df):
    """
    :return: change: Series
    """
    diffs = df['open'].diff(periods=1)
    shifts = df['open'].shift(1)
    change = diffs / shifts
    return change

def get_ret(df):
    """
    :return: ret: Series
    """
    change = get_change(df)
    ret = 1 + change
    return ret

def get_accum_ret(df, signal):
    """
    :param signal: Series(Boolean)
    :return: ret_by_signal: float64
    """
    ret_by_signal = 1
    ret_list = get_ret_list(df, signal)
    for ret in ret_list:
        ret_by_signal *= ret
    return ret_by_signal