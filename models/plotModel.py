import pandas as pd
import numpy as np
from production.codes.models import mt5Model, coinModel, maModel
from production.codes.utils import maths
from production.codes.models.backtestModel import returnModel, signalModel, statModel

def _get_format_plot_data(df=None, hist=None, text=None, equation=None, height=2):
    """
    :param df: pd.DataFrame
    :param hist: pd.Series
    :param text: str
    :param equation: str
    :param height: int
    :return: dictionary
    """
    plt_data = {}
    plt_data['df'] = df
    plt_data['hist'] = hist
    plt_data['text'] = text
    plt_data['equation'] = equation
    plt_data['height'] = height
    return plt_data

def get_total_height(plt_datas):
    # graph proportion
    total_height = 0
    for plt_data in plt_datas.values():
        total_height += plt_data['height']
    return total_height

def get_coin_NN_plot_title(start, end, timeframe_str):
    start_str = mt5Model.get_time_string(start)
    if end != None:
        end_str = mt5Model.get_time_string(end)
    else:
        end_str = mt5Model.get_current_time_string()
    title = "{} : {}, {}".format(start_str, end_str, timeframe_str)
    return title

def get_coin_NN_plot_image_name(dt_str, symbols, episode):
    symbols_str = ''
    for symbol in symbols:
        symbols_str += '_' + symbol
    name = "{}-{}-episode-{}.jpg".format(dt_str, episode, symbols_str)
    return name

def get_coin_NN_plt_datas(Prices, coefficient_vector, upper_th, lower_th, z_score_mean_window, z_score_std_window):
    """
    :param Prices: collections.nametuple object
    :param coefficient_vector: np.array
    :param upper_th: float
    :param lower_th: float
    :param z_score_rolling_mean_window: int
    :return: nested dictionary
    """
    # debug
    df_debug = pd.DataFrame(index=Prices.c.index)

    # prepare
    coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector, z_score_mean_window, z_score_std_window) # get_coin_data() can work for coinNN and coin
    long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, upper_th, lower_th)
    stats = statModel.get_stats(Prices, long_signal, short_signal, coefficient_vector)
    plt_datas = {}
    df_debug = pd.concat([df_debug, coin_data, long_signal, short_signal], axis=1)  #------------ DEBUG -------------

    # 1 graph: real and predict
    df = pd.concat([coin_data['real'], coin_data['predict']], axis=1)
    adf_result_text = get_ADF_text_result(coin_data['spread'].values)
    equation = get_coin_NN_equation_text(Prices.c.columns, coefficient_vector)
    plt_datas[0] = _get_format_plot_data(df=df, text=adf_result_text, equation=equation)
    df_debug = pd.concat([df_debug, df], axis=1)    #------------ DEBUG -------------

    # 2 graph: spread
    df = pd.DataFrame(coin_data['spread'], index=Prices.c.index)
    plt_datas[1] = _get_format_plot_data(df=df)
    df_debug = pd.concat([df_debug, df], axis=1)    #------------ DEBUG -------------

    # 3 graph: return for long and short
    df = pd.DataFrame(index=Prices.c.index)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector, long_mode=True)
    df_debug = pd.concat([df_debug, ret], axis=1)   #------------ DEBUG -------------
    df["long_accum_ret"] = returnModel.get_accum_ret(ret, long_signal)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector, long_mode=False)
    df_debug = pd.concat([df_debug, ret], axis=1)   #------------ DEBUG -------------
    df["short_accum_ret"] = returnModel.get_accum_ret(ret, short_signal)
    text = get_stat_text_condition(stats, 'ret')
    plt_datas[2] = _get_format_plot_data(df=df, text=text)
    df_debug = pd.concat([df_debug, df], axis=1)    #------------ DEBUG -------------

    # 4 graph: earning
    df = pd.DataFrame(index=Prices.c.index)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=True)
    df_debug = pd.concat([df_debug, earning], axis=1)   #------------ DEBUG -------------
    df["long_accum_earning"] = returnModel.get_accum_earning(earning, long_signal)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=False)
    df_debug = pd.concat([df_debug, earning], axis=1)   #------------ DEBUG -------------
    df["short_accum_earning"] = returnModel.get_accum_earning(earning, short_signal)
    text = get_stat_text_condition(stats, 'earning')
    plt_datas[3] = _get_format_plot_data(df=df, text=text)
    df_debug = pd.concat([df_debug, df], axis=1)        #------------ DEBUG -------------

    # 5 graph: z-score
    df = pd.DataFrame(coin_data['z_score'], index=Prices.c.index)
    plt_datas[4] = _get_format_plot_data(df=df)
    df_debug = pd.concat([df_debug, df], axis=1)        #------------ DEBUG -------------

    # 6 graph: ret histogram for long
    long_earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, coefficient_vector=coefficient_vector,
                                             signal=long_signal, long_mode=True)
    plt_datas[5] = _get_format_plot_data(hist=pd.Series(long_earning_list, name='long earning'))

    # 7 graph: ret histogram for short
    short_earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, coefficient_vector=coefficient_vector,
                                              signal=short_signal, long_mode=False)
    plt_datas[6] = _get_format_plot_data(hist=pd.Series(short_earning_list, name='short earning'))

    return plt_datas

def get_ma_plt_datas(Prices, long_param, short_param, limit_unit):
    """
    :param Prices: collections nametuples
    :param long_param: dict ['fast', 'slow']
    :param short_param: dict ['fast', 'slow']
    :return:
    """
    long_ma_data = maModel.get_ma_data(Prices.c, long_param['fast'], long_param['slow'])
    short_ma_data = maModel.get_ma_data(Prices.c, short_param['fast'], short_param['slow'])
    long_signal, short_signal = signalModel.get_movingAverage_signal(long_ma_data, short_ma_data, limit_unit=limit_unit)
    stats = statModel.get_stats(Prices, long_signal, short_signal, coefficient_vector=np.array([]))
    plt_datas = {}

    # 1 graph: close price, fast ma,  slow ma (long)
    df = pd.concat([Prices.c, long_ma_data['fast'], long_ma_data['slow']], axis=1)
    text = 'Long: \n  fast: {}\n  slow: {}'.format(long_param['fast'], long_param['slow'])
    plt_datas[0] = _get_format_plot_data(df=df, text=text)

    # 2 graph: close price, fast ma,  slow ma (short)
    df = pd.concat([Prices.c, short_ma_data['fast'], short_ma_data['slow']], axis=1)
    text = 'Short: \n  fast: {}\n  slow: {}'.format(short_param['fast'], short_param['slow'])
    plt_datas[1] = _get_format_plot_data(df=df, text=text)

    # 3 graph: ret (long and short)
    df = pd.DataFrame(index=Prices.c.index)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector=np.array([]), long_mode=True)
    df["long_accum_ret"] = returnModel.get_accum_ret(ret, long_signal)
    ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector=np.array([]), long_mode=False)
    df["short_accum_ret"] = returnModel.get_accum_ret(ret, short_signal)
    text = get_stat_text_condition(stats, 'ret')
    plt_datas[2] = _get_format_plot_data(df=df, text=text)

    # 4 graph: earning (long and short)
    df = pd.DataFrame(index=Prices.c.index)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=True)
    df["long_accum_earning"] = returnModel.get_accum_earning(earning, long_signal)
    earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=False)
    df["short_accum_earning"] = returnModel.get_accum_earning(earning, short_signal)
    text = get_stat_text_condition(stats, 'earning')
    plt_datas[3] = _get_format_plot_data(df=df, text=text)

    # 5 graph: ret histogram for long
    long_earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]),
                                             signal=long_signal, long_mode=True)
    plt_datas[4] = _get_format_plot_data(hist=pd.Series(long_earning_list, name='long earning'))

    # 6 graph: ret histogram for short
    short_earning_list = returnModel.get_earning_list(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]),
                                              signal=short_signal, long_mode=False)
    plt_datas[5] = _get_format_plot_data(hist=pd.Series(short_earning_list, name='short earning'))

    return plt_datas

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

def get_coin_NN_equation_text(symbols, coefficient_vector):
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


