import numpy as np
from production.codes.backtest import indexModel, returnModel
from production.codes.utils import tools

# def get_action_total(signal):
#     """
#     :param signal: pd.Series(Boolean)
#     :return: int
#     """
#     start, end = indexModel.get_start_end_index(signal.reset_index(drop=True))
#
#     return len(start)

def get_stat(ret_list, earning_list):
    """
    :param ret_list: []
    :param earning_list: []
    :return:
    """
    stat = {}
    total_ret, total_earning = returnModel.get_total_ret_earning(ret_list, earning_list)
    # earning
    stat['earning'] = {}
    stat['earning']['count'] = len(earning_list)
    stat['earning']["accuracy"] = tools.get_accuracy(earning_list, 0.0) # calculate the accuracy separately, note 46a
    stat['earning']["total"] = total_earning
    stat['earning']["mean"] = np.mean(earning_list)
    stat['earning']["max"] = np.max(earning_list)
    stat['earning']["min"] = np.min(earning_list)
    stat['earning']["std"] = np.std(earning_list)

    # return
    stat['ret'] = {}
    stat['ret']['count'] = len(earning_list)
    stat['ret']["accuracy"] = tools.get_accuracy(ret_list, 1.0) # calculate the accuracy separately, note 46a
    stat['ret']["total"] = total_ret
    stat['ret']["mean"] = np.mean(ret_list)
    stat['ret']["max"] = np.max(ret_list)
    stat['ret']["min"] = np.min(ret_list)
    stat['ret']["std"] = np.std(ret_list)

    return stat

def get_stats(long_ret_list, long_earning_list, short_ret_list, short_earning_list):
    """
    get stats both for long and short
    :param Prices: collections nametuple object
    :param long_signal: pd.Series
    :param short_signal: pd.Series
    :param coefficient_vector: np.array, raw coefficient
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: dictionary
    """
    stats = {}
    stats['long'] = get_stat(long_ret_list, long_earning_list)
    stats['short'] = get_stat(short_ret_list, short_earning_list)
    return stats

