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

def get_plot_title(start, end, timeframe_str):
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

def get_coin_NN_plt_datas(Prices, coefficient_vector, upper_th, lower_th, z_score_mean_window, z_score_std_window, slsp=(0,0), debug_file='debug.csv', debug=False):
    """
    :param Prices: collections.nametuple object
    :param coefficient_vector: np.array
    :param upper_th: float
    :param lower_th: float
    :param z_score_mean_window: int
    :param z_score_std_window: int
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :param debug_file: str
    :param debug: Boolean
    :return: nested dictionary
    """
    # prepare
    coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector, z_score_mean_window, z_score_std_window) # get_coin_data() can work for coinNN and coin
    long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, upper_th, lower_th)
    stats = statModel.get_stats(Prices, long_signal, short_signal, coefficient_vector)
    stats_slsp = statModel.get_stats(Prices, long_signal, short_signal, coefficient_vector, slsp)
    plt_datas = {}

    # 1 graph: real and predict
    real_predict_df = pd.concat([coin_data['real'], coin_data['predict']], axis=1)
    adf_result_text = get_ADF_text_result(coin_data['spread'].values)
    equation = get_coin_NN_equation_text(Prices.c.columns, coefficient_vector)
    plt_datas[0] = _get_format_plot_data(df=real_predict_df, text=adf_result_text, equation=equation)

    # 2 graph: spread
    spread_df = pd.DataFrame(coin_data['spread'], index=Prices.c.index)
    plt_datas[1] = _get_format_plot_data(df=spread_df)

    # 3 graph: z-score
    z_df = pd.DataFrame(coin_data['z_score'], index=Prices.c.index)
    plt_datas[2] = _get_format_plot_data(df=z_df)

    # prepare data for graph 4 and 5
    long_ret, long_earning = returnModel.get_ret_earning(Prices.o, Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=True)
    short_ret, short_earning = returnModel.get_ret_earning(Prices.o, Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_mode=False)
    long_accum_ret, long_accum_earning = returnModel.get_accum_ret_earning(long_ret, long_earning, long_signal)
    short_accum_ret, short_accum_earning = returnModel.get_accum_ret_earning(short_ret, short_earning, short_signal)

    # 4 graph: return for long and short
    accum_ret_df = pd.DataFrame(index=Prices.c.index)
    accum_ret_df["long_accum_ret"] = long_accum_ret
    accum_ret_df["short_accum_ret"] = short_accum_ret
    text = get_stat_text_condition(stats, 'ret')
    plt_datas[3] = _get_format_plot_data(df=accum_ret_df, text=text)

    # 5 graph: earning for long and short
    accum_earning_df = pd.DataFrame(index=Prices.c.index)
    accum_earning_df["long_accum_earning"] = long_accum_earning
    accum_earning_df["short_accum_earning"] = short_accum_earning
    text = get_stat_text_condition(stats, 'earning')
    plt_datas[4] = _get_format_plot_data(df=accum_earning_df, text=text)

    # prepare data for graph 6 and 7
    _, long_earning_list = returnModel.get_ret_earning_list(Prices.o, Prices.quote_exchg, Prices.ptDv,coefficient_vector=coefficient_vector, signal=long_signal, long_mode=True)
    _, short_earning_list = returnModel.get_ret_earning_list(Prices.o, Prices.quote_exchg, Prices.ptDv,coefficient_vector=coefficient_vector, signal=short_signal, long_mode=False)

    # 6 graph: earning histogram for long
    plt_datas[5] = _get_format_plot_data(hist=pd.Series(long_earning_list, name='long earning'))

    # 7 graph: earning histogram for short
    plt_datas[6] = _get_format_plot_data(hist=pd.Series(short_earning_list, name='short earning'))

    # prepare data for graph 6 and 7: ret and earning with stop-loss and stop-profit
    long_accum_ret_slsp, long_accum_earning_slsp = returnModel.get_accum_ret_earning(long_ret, long_earning, long_signal, slsp)
    short_accum_ret_slsp, short_accum_earning_slsp = returnModel.get_accum_ret_earning(short_ret, short_earning, short_signal, slsp)

    # 8 graph: ret with stop loss and stop profit for long and short
    accum_ret_slsp = pd.DataFrame(index=Prices.c.index)
    accum_ret_slsp['long_accum_ret_slsp'] = long_accum_ret_slsp
    accum_ret_slsp['short_accum_ret_slsp'] = short_accum_ret_slsp
    text = get_stat_text_condition(stats_slsp, 'ret')
    plt_datas[7] = _get_format_plot_data(df=accum_ret_slsp, text=text)

    # 9 graph: earning with stop loss and stop profit for long and short
    accum_earning_slsp = pd.DataFrame(index=Prices.c.index)
    accum_earning_slsp['long_accum_earning_slsp'] = long_accum_earning_slsp
    accum_earning_slsp['short_accum_earning_slsp'] = short_accum_earning_slsp
    text = get_stat_text_condition(stats_slsp, 'earning')
    plt_datas[8] = _get_format_plot_data(df=accum_earning_slsp, text=text)

    # prepare data for graph 10 and 11
    _, long_earning_list = returnModel.get_ret_earning_list(Prices.o, Prices.quote_exchg, Prices.ptDv, coefficient_vector=coefficient_vector, signal=long_signal, long_mode=True, slsp=slsp)
    _, short_earning_list = returnModel.get_ret_earning_list(Prices.o, Prices.quote_exchg, Prices.ptDv, coefficient_vector=coefficient_vector, signal=short_signal, long_mode=False, slsp=slsp)

    # 10 graph: earning histogram for long
    plt_datas[9] = _get_format_plot_data(hist=pd.Series(long_earning_list, name='long earning slsp'))

    # 11 graph: earning histogram for short
    plt_datas[10] = _get_format_plot_data(hist=pd.Series(short_earning_list, name='short earning slsp'))

    # ------------ DEBUG -------------
    if debug:
        df_debug = pd.DataFrame(index=Prices.c.index)
        df_debug = pd.concat([df_debug, Prices.o, Prices.quote_exchg, Prices.base_exchg, Prices.ptDv, coin_data, long_signal, short_signal,
                              long_ret, short_ret, accum_ret_df,
                              long_earning, short_earning, accum_earning_df,
                              long_accum_ret_slsp, short_accum_ret_slsp,
                              long_accum_earning_slsp, short_accum_earning_slsp], axis=1)
        df_debug.to_csv('C://Users//Chris//projects//210215_mt5//production//docs//1//debug//{}'.format(debug_file))

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
    long_ret = returnModel.get_ret(Prices.o, coefficient_vector=np.array([]), long_mode=True)
    df["long_accum_ret"] = returnModel.get_accum_ret(long_ret, long_signal)
    short_ret = returnModel.get_ret(Prices.o, coefficient_vector=np.array([]), long_mode=False)
    df["short_accum_ret"] = returnModel.get_accum_ret(short_ret, short_signal)
    text = get_stat_text_condition(stats, 'ret')
    plt_datas[2] = _get_format_plot_data(df=df, text=text)

    # 4 graph: earning (long and short)
    df = pd.DataFrame(index=Prices.c.index)
    long_earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=True)
    df["long_accum_earning"] = returnModel.get_accum_earning(long_earning, long_signal)
    short_earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=False)
    df["short_accum_earning"] = returnModel.get_accum_earning(short_earning, short_signal)
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

    # ------------ DEBUG -------------
    # long_ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector=np.array([]), long_mode=True)
    # short_ret = returnModel.get_ret(Prices.o, Prices.quote_exchg, coefficient_vector=np.array([]), long_mode=False)
    # long_earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=True)
    # short_earning = returnModel.get_earning(Prices.quote_exchg, Prices.ptDv, coefficient_vector=np.array([]), long_mode=False)

    return plt_datas

def get_spread_plt_datas(spreads):
    """
    :param spreads: pd.DataFrame
    :return: plt_data, nested dict
    """
    i = 0
    plt_data = {}
    for symbol in spreads.columns:
        plt_data[i] = _get_format_plot_data(df=pd.DataFrame(spreads[symbol]), text="mean: {:.2f} pt\nstd: {:.2f} pt".format(np.mean(spreads[symbol]), np.std(spreads[symbol]))) # np.mean(pd.Series) will ignore the nan value, note 56c
        i += 1
        plt_data[i] = _get_format_plot_data(hist=spreads[symbol])
        i += 1
    return plt_data

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

def get_setting_txt(setting_dict):
    setting = 'Setting: \n'
    for key, value in setting_dict.items():
        setting += "{}: {}\n".format(key, value)
    return setting

