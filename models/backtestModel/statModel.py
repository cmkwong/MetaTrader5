import numpy as np
from production.codes.models.backtestModel import returnModel, indexModel, dateModel, signalModel


def get_accuracy(open_price, signal):
    """
    :param open_price: pd.Series
    :param signal: pd.Series (Boolean)
    :return: float
    """
    rets = returnModel.get_ret_list(open_price, signal)
    accuracy = np.sum([c > 1 for c in rets]) / len(rets)
    return accuracy

def get_action_total(signal):
    """
    :param signal: pd.Series(Boolean)
    :return: int
    """
    start, end = indexModel.get_action_start_end_index(signal)

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

def get_stat(open_prices, earning, signal, coefficient_vector=1):
    """
    :param open_prices: pd.Series (There may be many different symbols of open prices)
    :param earning:
    :param signal:
    :param coefficient_vector:
    :return: stat, dictionary
    """
    stat = {}
    signal = signalModel.discard_head_signal(signal)
    signal = signalModel.discard_tail_signal(signal)
    if signal.sum() != 0:
        stat["count"] = get_action_total(signal)
        stat["accum_earning"] = returnModel.get_accum_earning(earning, signal)
        ret_arr = np.zeros((stat['count'], len(open_prices.columns)))
        for c, symbol in enumerate(open_prices):
            ret_list = returnModel.get_ret_list(open_prices[symbol], signal)
            ret_arr[:,c] = ret_list
        weight_factor = np.append(-1 * coefficient_vector[1:], 1) # long spread mode
        ret_overall = np.sum(ret_arr * weight_factor, axis=1)
        stat["accum_ret"] = np.sum(ret_overall, axis=0)
        stat["mean"] = np.mean(ret_overall)
        stat["max"] = np.max(ret_overall)
        stat["min"] = np.min(ret_overall)
        stat["std"] = np.std(ret_overall)
        stat["acc"] = np.sum([c > 1 for c in ret_overall]) / len(ret_overall)
    return stat


