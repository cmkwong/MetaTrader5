import pandas as pd
import numpy as np
import collections
import os

from production.codes.models import coinModel, maModel, timeModel
from production.codes.utils import maths
from production.codes.models.backtestModel import returnModel, signalModel, statModel, exchgModel

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

def _get_graph_data(Prices, long_signal, short_signal, coefficient_vector):
    Graph_Data = collections.namedtuple("Graph_Data", ["long_modify_exchg_q2d", "short_modify_exchg_q2d",
                                                       "long_ret", "long_earning",
                                                       "long_accum_ret", "long_accum_earning",
                                                       "short_ret", "short_earning",
                                                       "short_accum_ret","short_accum_earning",
                                                       "long_ret_list", "long_earning_list",
                                                       "short_ret_list","short_earning_list",
                                                       "stats"])
    # prepare q2d
    long_modify_exchg_q2d = exchgModel.get_exchg_by_signal(Prices.quote_exchg, long_signal)
    short_modify_exchg_q2d = exchgModel.get_exchg_by_signal(Prices.quote_exchg, short_signal)

    # prepare data for graph ret and earning
    long_ret, long_earning = returnModel.get_ret_earning(Prices.o, Prices.o.shift(1), long_modify_exchg_q2d, Prices.ptDv, coefficient_vector, long_mode=True)
    long_ret_by_signal, long_earning_by_signal = returnModel.get_ret_earning_by_signal(long_ret, long_earning, long_signal)
    long_accum_ret, long_accum_earning = returnModel.get_accum_ret_earning(long_ret_by_signal, long_earning_by_signal)

    short_ret, short_earning = returnModel.get_ret_earning(Prices.o, Prices.o.shift(1), short_modify_exchg_q2d, Prices.ptDv, coefficient_vector, long_mode=False)
    short_ret_by_signal, short_earning_by_signal = returnModel.get_ret_earning_by_signal(short_ret, short_earning, short_signal)
    short_accum_ret, short_accum_earning = returnModel.get_accum_ret_earning(short_ret_by_signal, short_earning_by_signal)

    # prepare data for graph histogram
    long_ret_list, long_earning_list = returnModel.get_ret_earning_list(long_ret_by_signal, long_earning_by_signal, long_signal)
    short_ret_list, short_earning_list = returnModel.get_ret_earning_list(short_ret_by_signal, short_earning_by_signal, short_signal)

    # prepare stat
    stats = statModel.get_stats(long_ret_list, long_earning_list, short_ret_list, short_earning_list)

    # assigned to Graph_Data
    Graph_Data.long_modify_exchg_q2d, Graph_Data.short_modify_exchg_q2d = long_modify_exchg_q2d, short_modify_exchg_q2d
    Graph_Data.long_ret, Graph_Data.long_earning = long_ret, long_earning
    Graph_Data.long_accum_ret, Graph_Data.long_accum_earning = long_accum_ret, long_accum_earning
    Graph_Data.short_ret, Graph_Data.short_earning = short_ret, short_earning
    Graph_Data.short_accum_ret, Graph_Data.short_accum_earning = short_accum_ret, short_accum_earning
    Graph_Data.long_ret_list, Graph_Data.long_earning_list = long_ret_list, long_earning_list
    Graph_Data.short_ret_list, Graph_Data.short_earning_list = short_ret_list, short_earning_list
    Graph_Data.stats = stats
    return Graph_Data

def get_total_height(plt_datas):
    # graph proportion
    total_height = 0
    for plt_data in plt_datas.values():
        total_height += plt_data['height']
    return total_height

def get_plot_title(start, end, timeframe_str, local):
    # end_str
    start_str = timeModel.get_time_string(start)
    if end != None:
        end_str = timeModel.get_time_string(end)
    else:
        end_str = timeModel.get_current_time_string()
    # local/mt5
    if local:
        source = 'local'
    else:
        source = 'mt5'
    title = "{} : {}, timeframe={}, source={}".format(start_str, end_str, timeframe_str, source)
    return title

def get_coin_NN_plot_image_name(dt_str, symbols, episode):
    symbols_str = ''
    for symbol in symbols:
        symbols_str += '_' + symbol
    name = "{}-{}-episode-{}.jpg".format(dt_str, episode, symbols_str)
    return name

def get_coin_NN_plt_datas(Prices, min_Prices, coefficient_vector, upper_th, lower_th, z_score_mean_window, z_score_std_window, close_change=0, slsp=None, timeframe=None,
                          debug_path='', debug_file='debug.csv', debug=False):
    """
    :param Prices: collections.nametuple object
    :param coefficient_vector: np.array
    :param upper_th: np.float
    :param lower_th: np.float
    :param z_score_mean_window: int
    :param z_score_std_window: int
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :param timeframe: needed when calculating slsp
    :param debug_file: str
    :param debug: Boolean
    :return: nested dictionary
    """
    # -------------------------------------------------------------------- standard --------------------------------------------------------------------
    # prepare signal
    dependent_variable = Prices.c
    if close_change == 1:
        dependent_variable = Prices.cc
    coin_data = coinModel.get_coin_data(dependent_variable, coefficient_vector, z_score_mean_window, z_score_std_window)  # get_coin_data() can work for coinNN and coin
    long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, upper_th, lower_th)
    # Get Graph Data
    Graph_Data = _get_graph_data(Prices, long_signal, short_signal, coefficient_vector)

    # -------------------------------------------------------------------- standard graph --------------------------------------------------------------------
    plt_datas = {}
    # 1 graph: real and predict
    real_predict_df = pd.concat([coin_data['real'], coin_data['predict']], axis=1)
    adf_result_text = get_ADF_text_result(coin_data['spread'].values)
    equation = get_coin_NN_equation_text(dependent_variable.columns, coefficient_vector)
    plt_datas[0] = _get_format_plot_data(df=real_predict_df, text=adf_result_text, equation=equation)

    # 2 graph: spread
    spread_df = pd.DataFrame(coin_data['spread'], index=dependent_variable.index)
    plt_datas[1] = _get_format_plot_data(df=spread_df)

    # 3 graph: z-score
    z_df = pd.DataFrame(coin_data['z_score'], index=dependent_variable.index)
    plt_datas[2] = _get_format_plot_data(df=z_df)

    # 4 graph: return for long and short
    accum_ret_df = pd.DataFrame(index=dependent_variable.index)
    accum_ret_df["long_accum_ret"] = Graph_Data.long_accum_ret
    accum_ret_df["short_accum_ret"] = Graph_Data.short_accum_ret
    text = get_stat_text_condition(Graph_Data.stats, 'ret')
    plt_datas[3] = _get_format_plot_data(df=accum_ret_df, text=text)

    # 5 graph: earning for long and short
    accum_earning_df = pd.DataFrame(index=dependent_variable.index)
    accum_earning_df["long_accum_earning"] = Graph_Data.long_accum_earning
    accum_earning_df["short_accum_earning"] = Graph_Data.short_accum_earning
    text = get_stat_text_condition(Graph_Data.stats, 'earning')
    plt_datas[4] = _get_format_plot_data(df=accum_earning_df, text=text)

    # 6 graph: earning histogram for long
    plt_datas[5] = _get_format_plot_data(hist=pd.Series(Graph_Data.long_earning_list, name='long earning'))

    # 7 graph: earning histogram for short
    plt_datas[6] = _get_format_plot_data(hist=pd.Series(Graph_Data.short_earning_list, name='short earning'))

    # ------------ DEBUG -------------
    df_debug = pd.DataFrame(index=Prices.o.index)
    df_debug = pd.concat([df_debug, Prices.o, Graph_Data.long_modify_exchg_q2d, Graph_Data.short_modify_exchg_q2d, Prices.ptDv, coin_data,
                          long_signal, short_signal,
                          Graph_Data.long_ret, Graph_Data.short_ret, accum_ret_df,
                          Graph_Data.long_earning, Graph_Data.short_earning, accum_earning_df
                          ], axis=1)

    # -------------------------------------------------------------------- slsp --------------------------------------------------------------------
    if len(min_Prices) != 0:
        # prepare minute data for slsp part
        long_min_signal, short_min_signal = signalModel.get_resoluted_signal(long_signal, min_Prices.quote_exchg.index), signalModel.get_resoluted_signal(short_signal, min_Prices.quote_exchg.index)
        long_modify_min_exchg_q2d = exchgModel.get_exchg_by_signal(min_Prices.quote_exchg, long_signal)
        short_modify_min_exchg_q2d = exchgModel.get_exchg_by_signal(min_Prices.quote_exchg, short_signal)
        long_min_ret, long_min_earning = returnModel.get_ret_earning(min_Prices.o, min_Prices.o.shift(1), long_modify_min_exchg_q2d, min_Prices.ptDv, coefficient_vector, long_mode=True)
        short_min_ret, short_min_earning = returnModel.get_ret_earning(min_Prices.o, min_Prices.o.shift(1), short_modify_min_exchg_q2d, min_Prices.ptDv, coefficient_vector, long_mode=False)

        # prepare data for graph 8 and 9: ret and earning with stop-loss and stop-profit
        long_ret_by_signal_slsp, long_earning_by_signal_slsp = returnModel.get_ret_earning_by_signal(Graph_Data.long_ret, Graph_Data.long_earning, long_signal, long_min_ret, long_min_earning, slsp, timeframe)
        long_accum_ret_slsp, long_accum_earning_slsp = returnModel.get_accum_ret_earning(long_ret_by_signal_slsp, long_earning_by_signal_slsp)
        short_ret_by_signal_slsp, short_earning_by_signal_slsp = returnModel.get_ret_earning_by_signal(Graph_Data.short_ret, Graph_Data.short_earning, short_signal, short_min_ret, short_min_earning, slsp, timeframe)
        short_accum_ret_slsp, short_accum_earning_slsp = returnModel.get_accum_ret_earning(short_ret_by_signal_slsp, short_earning_by_signal_slsp)

        # prepare data for graph 10 and 11
        long_ret_list_slsp, long_earning_list_slsp = returnModel.get_ret_earning_list(long_ret_by_signal_slsp, long_earning_by_signal_slsp, long_signal)
        short_ret_list_slsp, short_earning_list_slsp = returnModel.get_ret_earning_list(short_ret_by_signal_slsp, short_earning_by_signal_slsp, short_signal)

        # prepare stat
        stats_slsp = statModel.get_stats(long_ret_list_slsp, long_earning_list_slsp, short_ret_list_slsp, short_earning_list_slsp)

        # -------------------------------------------------------------------- slsp graph --------------------------------------------------------------------
        # 8 graph: ret with stop loss and stop profit for long and short
        accum_ret_slsp = pd.DataFrame(index=dependent_variable.index)
        accum_ret_slsp['long_accum_ret_slsp'] = long_accum_ret_slsp
        accum_ret_slsp['short_accum_ret_slsp'] = short_accum_ret_slsp
        text = get_stat_text_condition(stats_slsp, 'ret')
        plt_datas[7] = _get_format_plot_data(df=accum_ret_slsp, text=text)

        # 9 graph: earning with stop loss and stop profit for long and short
        accum_earning_slsp = pd.DataFrame(index=dependent_variable.index)
        accum_earning_slsp['long_accum_earning_slsp'] = long_accum_earning_slsp
        accum_earning_slsp['short_accum_earning_slsp'] = short_accum_earning_slsp
        text = get_stat_text_condition(stats_slsp, 'earning')
        plt_datas[8] = _get_format_plot_data(df=accum_earning_slsp, text=text)

        # 10 graph: earning histogram for long
        plt_datas[9] = _get_format_plot_data(hist=pd.Series(long_earning_list_slsp, name='long earning slsp'))

        # 11 graph: earning histogram for short
        plt_datas[10] = _get_format_plot_data(hist=pd.Series(short_earning_list_slsp, name='short earning slsp'))

        # ------------ slsp DEBUG -------------
        # concat more data if slsp available
        df_debug = pd.concat([df_debug,
                              long_accum_ret_slsp, short_accum_ret_slsp,
                              long_accum_earning_slsp, short_accum_earning_slsp], axis=1)
        # for minute data
        range = [150000, 200000]
        df_min_debug = pd.DataFrame(index=min_Prices.o.iloc[range[0]:range[1]].index)
        df_min_debug = pd.concat([df_min_debug, min_Prices.o.iloc[range[0]:range[1]], long_modify_min_exchg_q2d.iloc[range[0]:range[1]],
                                  short_modify_min_exchg_q2d.iloc[range[0]:range[1]], min_Prices.ptDv.iloc[range[0]:range[1]],
                                  long_min_signal.iloc[range[0]:range[1]], short_min_signal.iloc[range[0]:range[1]],
                                  long_min_ret.iloc[range[0]:range[1]], short_min_ret.iloc[range[0]:range[1]],
                                  long_min_earning.iloc[range[0]:range[1]], short_min_earning.iloc[range[0]:range[1]]
                                  ], axis=1)
        if debug:
            df_min_debug.to_csv(os.path.join(debug_path, "min_"+debug_file))
    if debug:
        df_debug.to_csv(os.path.join(debug_path, debug_file))

    return plt_datas

def get_ma_plt_datas(Prices, long_param, short_param, limit_unit, debug_path, debug_file, debug=False):
    """
    :param Prices: collections nametuples
    :param long_param: dict ['fast', 'slow']
    :param short_param: dict ['fast', 'slow']
    :return:
    """
    # prepare
    long_ma_data = maModel.get_ma_data(Prices.c, long_param['fast'], long_param['slow'])
    short_ma_data = maModel.get_ma_data(Prices.c, short_param['fast'], short_param['slow'])
    long_signal, short_signal = signalModel.get_movingAverage_signal(long_ma_data, short_ma_data, limit_unit=limit_unit)
    # Get Graph Data
    Graph_Data = _get_graph_data(Prices, long_signal, short_signal, coefficient_vector=np.array([]))

    # -------------------------------------------------------------------- standard graph --------------------------------------------------------------------
    plt_datas = {}
    # 1 graph: close price, fast ma,  slow ma (long)
    long_ma_df = pd.concat([Prices.c, long_ma_data['fast'], long_ma_data['slow']], axis=1)
    text = 'Long: \n  fast: {}\n  slow: {}'.format(long_param['fast'], long_param['slow'])
    plt_datas[0] = _get_format_plot_data(df=long_ma_df, text=text)

    # 2 graph: close price, fast ma,  slow ma (short)
    short_ma_df = pd.concat([Prices.c, short_ma_data['fast'], short_ma_data['slow']], axis=1)
    text = 'Short: \n  fast: {}\n  slow: {}'.format(short_param['fast'], short_param['slow'])
    plt_datas[1] = _get_format_plot_data(df=short_ma_df, text=text)

    # 3 graph: ret (long and short)
    accum_ret_df = pd.DataFrame(index=Prices.c.index)
    accum_ret_df["long_accum_ret"] = Graph_Data.long_accum_ret
    accum_ret_df["short_accum_ret"] = Graph_Data.short_accum_ret
    text = get_stat_text_condition(Graph_Data.stats, 'ret')
    plt_datas[2] = _get_format_plot_data(df=accum_ret_df, text=text)

    # 4 graph: earning (long and short)
    accum_earning_df = pd.DataFrame(index=Prices.c.index)
    accum_earning_df["long_accum_earning"] = Graph_Data.long_accum_earning
    accum_earning_df["short_accum_earning"] = Graph_Data.short_accum_earning
    text = get_stat_text_condition(Graph_Data.stats, 'earning')
    plt_datas[3] = _get_format_plot_data(df=accum_earning_df, text=text)

    # 5 graph: ret histogram for long
    plt_datas[4] = _get_format_plot_data(hist=pd.Series(Graph_Data.long_ret_list, name='long earning'))

    # 6 graph: ret histogram for short
    plt_datas[5] = _get_format_plot_data(hist=pd.Series(Graph_Data.short_earning_list, name='short earning'))

    # ------------ DEBUG -------------
    df_debug = pd.DataFrame(index=Prices.o.index)
    df_debug = pd.concat([df_debug, Prices.o, Graph_Data.long_modify_exchg_q2d, Graph_Data.short_modify_exchg_q2d, Prices.ptDv,
                          long_ma_data, short_ma_data,
                          long_signal, short_signal,
                          Graph_Data.long_ret, Graph_Data.short_ret, accum_ret_df,
                          Graph_Data.long_earning, Graph_Data.short_earning, accum_earning_df
                          ], axis=1)
    if debug:
        df_debug.to_csv(os.path.join(debug_path, debug_file))
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
        # for nested dictionary (Second level)
        if type(value) == dict:
            setting += "{}: \n".format(key)
            for k, v in value.items():
                setting += "  {}: {}\n".format(k, v)
        # if only one level of dictionary
        else:
            setting += "{}: {}\n".format(key, value)
    return setting

