import numpy as np
from production.codes.models.backtestModel import returnModel, indexModel, dateModel, signalModel

def get_accuracy(rets):
    accuracy = np.sum([c > 1 for c in rets]) / len(rets)
    return accuracy

# def get_accuracy2(open_price, signal):
#     """
#     :param open_price: pd.Series
#     :param signal: pd.Series (Boolean)
#     :return: float
#     """
#     rets = returnModel.get_ret_list(open_price, signal)
#     accuracy = np.sum([c > 1 for c in rets]) / len(rets)
#     return accuracy

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
    :param open_prices: pd.Dataframe
    :param earning:
    :param signal: pd.Series
    :param coefficient_vector:
    :return:
    """
    stat = {}
    signal = signalModel.discard_head_signal(signal)
    signal = signalModel.discard_tail_signal(signal)
    modified_coefficient_vector = signalModel.get_modify_coefficient_vector(long_mode, coefficient_vector)
    if signal.sum() != 0:
        # total number of trade
        stat["count"] = get_action_total(signal)

        # earning
        earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, modified_coefficient_vector, signal)
        stat["accum_earning"] = returnModel.get_accum_earning(earning_list)
        stat["mean_earning"] = np.mean(earning_list)
        stat["max_earning"] = np.max(earning_list)
        stat["min_earning"] = np.min(earning_list)
        stat["std_earning"] = np.std(earning_list)

        # return
        ret_list = returnModel.get_ret_list(Prices.o, Prices.quote_exchg, modified_coefficient_vector, signal, long_mode)
        stat["accum_ret"] = returnModel.get_accum_ret(ret_list)
        stat["mean_ret"] = np.mean(ret_list)
        stat["max_ret"] = np.max(ret_list)
        stat["min_ret"] = np.min(ret_list)
        stat["std_ret"] = np.std(ret_list)

        stat["accuracy"] = get_accuracy(ret_list)
    return stat

# def get_stat2(df, signal):
#     """
#     :return: stat dictionary
#     """
#     stat = {}
#     if signal.sum() != 0:
#         stat["count"] = get_action_total(signal)
#         stat["accum"] = returnModel.get_accum_ret(df, signal)
#         stat["mean"] = np.mean(returnModel.get_ret_list(df, signal))
#         stat["max"] = np.max(returnModel.get_ret_list(df, signal))
#         stat["min"] = np.min(returnModel.get_ret_list(df, signal))
#         stat["std"] = np.std(returnModel.get_ret_list(df, signal))
#         stat["acc"] = get_accuracy(df, signal)
#     return stat
#
# def get_stat3(open_prices, earning, signal, coefficient_vector):
#     """
#     :param open_prices: pd.Series (There may be many different symbols of open prices)
#     :param earning:
#     :param signal:
#     :param coefficient_vector:
#     :return: stat, dictionary
#     """
#     stat = {}
#     signal = signalModel.discard_head_signal(signal)
#     signal = signalModel.discard_tail_signal(signal)
#     if signal.sum() != 0:
#         stat["count"] = get_action_total(signal)
#         stat["accum_earning"] = returnModel.get_accum_earning(earning, signal)
#         ret_arr = np.zeros((stat['count'], len(open_prices.columns)))
#         for c, symbol in enumerate(open_prices):
#             ret_list = returnModel.get_ret_list(open_prices[symbol], signal)
#             ret_arr[:,c] = ret_list
#         weight_factor = np.append(-1 * coefficient_vector[1:], 1) # long spread mode
#         ret_overall = np.sum(ret_arr * weight_factor, axis=1)
#         stat["accum_ret"] = np.sum(ret_overall, axis=0)
#         stat["mean"] = np.mean(ret_overall)
#         stat["max"] = np.max(ret_overall)
#         stat["min"] = np.min(ret_overall)
#         stat["std"] = np.std(ret_overall)
#         stat["acc"] = np.sum([c > 1 for c in ret_overall]) / len(ret_overall)
#     return stat


