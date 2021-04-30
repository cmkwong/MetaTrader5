import numpy as np
from production.codes.models import returnModel, indexModel, dateModel

def get_accuracy(df, signal):
    ret_list = returnModel.get_ret_list(df, signal)
    accuracy = np.sum([r > 1 for r in ret_list]) / len(ret_list)
    return accuracy

def get_action_total(signal):
    """
    :param signal: Series(Boolean)
    :return: int
    """
    start, end = indexModel.get_action_start_end_index(signal)

    return len(start)

def get_action_detail(df, signal):
    """
    :param signal: Series
    :return: action_details: dictionary
    """
    action_details = {}
    start_dates, end_dates = dateModel.get_action_date(df, signal)
    ret_list = returnModel.get_ret_list(df, signal)
    for s, e, r in zip(start_dates, end_dates, ret_list):
        key = s + '-' + e
        action_details[key] = r
    return action_details

def get_stat(df, signal):
    """
    :return: stat dictionary
    """
    stat = {}
    if signal.sum() != 0:
        stat["count"] = get_action_total(signal)
        stat["accum"] = returnModel.get_accum_ret(df, signal)
        stat["mean"] = np.mean(returnModel.get_ret_list(df, signal))
        stat["max"] = np.max(returnModel.get_ret_list(df, signal))
        stat["min"] = np.min(returnModel.get_ret_list(df, signal))
        stat["std"] = np.std(returnModel.get_ret_list(df, signal))
        stat["acc"] = get_accuracy(df, signal)
    return stat


