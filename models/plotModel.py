import pandas as pd
from production.codes.models import mt5Model, coinModel
from production.codes.utils import maths
from production.codes.models.backtestModel import returnModel

def get_coin_plot_title(start, end, timeframe_str):
    start_str = mt5Model.get_time_string(start)
    if end != None:
        end_str = mt5Model.get_time_string(end)
    else:
        end_str = mt5Model.get_current_time_string()
    title = "{} : {}, {}".format(start_str, end_str, timeframe_str)
    return title

def get_coin_plot_image_name(dt_str, symbols, episode):
    symbols_str = ''
    for symbol in symbols:
        symbols_str += '_' + symbol
    name = "{}-{}-episode-{}.jpg".format(dt_str, episode, symbols_str)
    return name

def get_coin_plt_data(Prices, long_signal, short_signal, coefficient_vector, stats):
    """
    :param Prices:
    :param long_signal:
    :param short_signal:
    :param coefficient_vector:
    :param stats: nametuple object included for long and short signal
    :return:
    """
    coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector)
    plt_data = {}

    # first graph: real and predict
    plt_data[0] = {}
    df = pd.concat([coin_data['real'], coin_data['predict']], axis=1)
    adf_result_text = get_ADF_text_result(coin_data['spread'].values)
    plt_data[0]['df'] = df
    plt_data[0]['text'] = adf_result_text
    plt_data[0]['equation'] = get_coin_equation_text(Prices.c.columns, coefficient_vector)

    # second graph: spread
    plt_data[1] = {}
    plt_data[1]['df'] = pd.DataFrame(coin_data['spread'], index=Prices.c.index)
    plt_data[1]['text'] = None
    plt_data[1]['equation'] = None

    # third graph: return
    plt_data[2] = {}
    df = pd.DataFrame(index=Prices.c.index)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector, long_mode=True)
    df["long_accum_ret"] = returnModel.get_accum_ret(ret, long_signal)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector, long_mode=False)
    df["short_accum_ret"] = returnModel.get_accum_ret(ret, short_signal)
    plt_data[2]['df'] = df
    plt_data[2]['text'] = get_stat_text_condition(stats, 'ret')
    plt_data[2]['equation'] = None

    # fourth graph: earning
    plt_data[3] = {}
    df = pd.DataFrame(index=Prices.c.index)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=True)
    df["long_accum_earning"] = returnModel.get_accum_earning(earning, long_signal)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=False)
    df["short_accum_earning"] = returnModel.get_accum_earning(earning, short_signal)
    plt_data[3]['df'] = df
    plt_data[3]['text'] = get_stat_text_condition(stats, 'earning')
    plt_data[3]['equation'] = None

    # fifth graph: z-score
    plt_data[4] = {}
    plt_data[4]['df'] = pd.DataFrame(coin_data['z_score'], index=Prices.c.index)
    plt_data[4]['text'] = None
    plt_data[4]['equation'] = None

    return plt_data

    # plt_df = pd.DataFrame(coin_data.values, index=coin_data.index, columns=coin_data.columns)
    # # long_mode
    # for long_mode in [True, False]:
    #     if long_mode:
    #         signal = long_signal
    #         mode_str = 'long'
    #     else:
    #         signal = short_signal
    #         mode_str = 'short'
    #     ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector, long_mode)
    #     accum_ret = returnModel.get_accum_ret(ret, signal).rename('{}_accum_ret'.format(mode_str)) # name: accum_ret
    #     earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode)
    #     accum_earning = returnModel.get_accum_earning(earning, signal).rename('{}_accum_earning'.format(mode_str)) # name: accum_earning
    #     plt_df = pd.concat([plt_df, accum_ret, accum_earning], axis=1)
    #
    # return plt_df

def append_all_df_debug(df_list):
    # [Prices.c, Prices.o, points_dff_values_df, coin_signal, int_signal, changes, ret_by_signal]
    prefix_names = ['open', 'pt_diff_values', 'q2d', 'b2d', 'ret', 'plt_data', 'signal', 'int_signal', 'earning', 'earning_by_signal']
    all_df = None
    for i, df in enumerate(df_list):
        df.columns = [(col_name + '_' + prefix_names[i]) for col_name in df.columns]
        if i == 0:
            all_df = pd.DataFrame(df.values, index=df.index, columns=df.columns)
        else:
            all_df = pd.concat([all_df, df], axis=1, join='inner')
    return all_df

def concatenate_plotting_df(train_plt_df, test_plt_df):
    """
    :param train_plt_df: pd.Dataframe
    :param test_plt_df: pd.Dataframe
    return df_plt: pd.Dataframe
    """
    df_plt = pd.concat(train_plt_df,test_plt_df)
    return df_plt

def get_ADF_text_result(spread):
    """
    :param spread: np.array
    :return:
    """
    text = ''
    result = maths.perform_ADF_test(spread)
    text += "The test statistic: {:.6f}\n".format(result.test_statistic)
    text += "The p-value: {:.6f}\n".format(result.pvalue)
    text += "The critical values: \n"
    for key, value in result.critical_values.items():
        text += "     {} = {:.6f}\n".format(key, value)
    return text

def get_stat_text_condition(stats, required_type):
    """
    :param stat: including the long and short stat
    :param required_type: str 'earning' / 'ret'
    :return: str, only for required type
    """
    txt = ''
    for mode, types in stats.items():  # long or short
        txt += "{}:\n".format(mode)
        txt += "count: {}\naccuracy: {:.5f}\n".format(types['count'], types['accuracy'])
        for type, stat in types.items():  # count, accuracy, (return / earning)
            if type == required_type:
                txt += "  {}:\n".format(type)
                for key, value in stat.items():  # stat dict
                    txt += "    {}:{:.5f}\n".format(key, value)
    return txt

def get_stats_text(stats):
    """
    :param stats: including the long and short stat
    :return: str
    """
    txt = ''
    for mode, types in stats.items():       # long or short
        txt += "{}:\n".format(mode)
        for type, stat in types:            # return or earning
            txt += "  {}:\n".format(type)
            for key, value in stat.items(): # stat dict
                txt += "    {}:{:.5f}\n".format(key, value)
    return txt

def get_coin_equation_text(symbols, coefficient_vector):
    """
    :param symbols: [str]
    :param coefficient_vector: np.array
    :return: str
    """
    coefficient_vector = coefficient_vector.reshape(-1, )
    equation = "{:.5f}".format(coefficient_vector[0])
    for i, symbol in enumerate(symbols[:-1]):
        if coefficient_vector[i+1] >= 0:
            equation += "+{:.5f}[{}]".format(coefficient_vector[i+1], symbol)
        else:
            equation += "{:.5f}[{}]".format(coefficient_vector[i+1], symbol)
    equation += " = [{}]".format(symbols[-1])
    return equation


