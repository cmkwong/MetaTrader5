import numpy as np
from production.codes.models.backtestModel import returnModel, indexModel
from production.codes.utils import tools

def get_action_total(signal):
    """
    :param signal: pd.Series(Boolean)
    :return: int
    """
    start, end = indexModel.get_action_start_end_index(signal.reset_index(drop=True))

    return len(start)

def get_action_detail(open_price, signal):
    """
    :param signal: Series
    :return: action_details: dictionary
    """
    action_details = {}
    start_indexs, end_indexs = indexModel.get_action_start_end_index(signal)
    rets = returnModel.get_ret_list(open_price, signal)
    for s, e, r in zip(start_indexs, end_indexs, rets):
        key = s + '-' + e
        action_details[key] = r
    return action_details

def get_stat(Prices, signal, coefficient_vector, long_mode=True):
    """
    :param Prices: collections nametuple object
    :param signal: pd.Series
    :param coefficient_vector: np.array, raw coefficient
    :param long_mode: Boolean
    :return:
    """
    stat = {}
    if signal.sum() != 0:
        # earning
        stat['earning'] = {}
        earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, coefficient_vector, signal, long_mode)
        stat['earning']['count'] = get_action_total(signal)
        stat['earning']["accuracy"] = tools.get_accuracy(earning_list, 0.0) # calculate the accuracy separately, note 46a
        stat['earning']["total"] = returnModel.get_total_earning(earning_list)
        stat['earning']["mean"] = np.mean(earning_list)
        stat['earning']["max"] = np.max(earning_list)
        stat['earning']["min"] = np.min(earning_list)
        stat['earning']["std"] = np.std(earning_list)

        # return
        stat['ret'] = {}
        ret_list = returnModel.get_ret_list(Prices.o, Prices.quote_exchg, coefficient_vector, signal, long_mode)
        stat['ret']['count'] = get_action_total(signal)
        stat['ret']["accuracy"] = tools.get_accuracy(ret_list, 1.0) # calculate the accuracy separately, note 46a
        stat['ret']["total"] = returnModel.get_total_ret(ret_list)
        stat['ret']["mean"] = np.mean(ret_list)
        stat['ret']["max"] = np.max(ret_list)
        stat['ret']["min"] = np.min(ret_list)
        stat['ret']["std"] = np.std(ret_list)

    return stat

def get_stats(Prices, long_signal, short_signal, coefficient_vector):
    """
    get stats both for long and short
    :param Prices: collections nametuple object
    :param long_signal: pd.Series
    :param short_signal: pd.Series
    :param coefficient_vector: np.array, raw coefficient
    :return: dictionary
    """
    stats = {}
    stats['long'] = get_stat(Prices, long_signal, coefficient_vector, long_mode=True)
    stats['short'] = get_stat(Prices, short_signal, coefficient_vector, long_mode=False)
    return stats

