import numpy as np
import pandas as pd
from production.codes.models import mt5Model, coinModel
from production.codes.utils import maths

def get_plot_title(start, end, timeframe_str):
    start_str = mt5Model.get_time_string(start)
    if end != None:
        end_str = mt5Model.get_time_string(end)
    else:
        end_str = mt5Model.get_current_time_string()
    title = "{} : {}, {}".format(start_str, end_str, timeframe_str)
    return title

def get_plot_image_name(dt_str, symbols, episode):
    symbols_str = ''
    for symbol in symbols:
        symbols_str += '_' + symbol
    name = "{}-{}-episode-{}.jpg".format(dt_str, episode, symbols_str)
    return name

def get_plotting_data(prices_matrix, model, seq_len):
    """
    :param prices_matrix: accept the train and test prices in array format
    :param model: torch model to get the predicted value
    :param data_options: dict
    :return: array for plotting
    """
    plt_data = {}
    plt_data['inputs'] = prices_matrix[:, :-1]
    plt_data['predict'] = model.get_predicted_arr(plt_data['inputs'], seq_len)
    plt_data['target'] = prices_matrix[:, -1]
    spread = plt_data['target'] - plt_data['predict']
    plt_data['spread'] = spread
    plt_data['z_score'] = maths.z_score_with_rolling_mean(spread, 10)
    return plt_data

def get_plotting_data_simple(prices_matrix, coefficient_vector):
    """
    :param prices_matrix: accept the train and test prices in array format
    :param coefficient_vector:
    :return:
    """
    plt_data = {}
    plt_data['inputs'] = prices_matrix[:, :-1]
    plt_data['predict'] = coinModel.get_predicted_arr(plt_data['inputs'], coefficient_vector)
    plt_data['target'] = prices_matrix[:, -1]
    spread = plt_data['target'] - plt_data['predict']
    plt_data['spread'] = spread
    plt_data['z_score'] = maths.z_score_with_rolling_mean(spread, 10)
    return plt_data

def concatenate_plotting_df(train_plt_data, test_plt_data, symbols):
    """
    :param train_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param test_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param symbols: [str]
    :return: dataframe
    """
    target = symbols[-1]
    df_plt = pd.DataFrame(columns=symbols)
    for c, symbol in enumerate(symbols[:-1]):
        df_plt[symbol] = np.concatenate((train_plt_data['inputs'][:,c], test_plt_data['inputs'][:,c]), axis=0).reshape(-1, )
    df_plt[target] = np.concatenate((train_plt_data['target'], test_plt_data['target']), axis=0)
    df_plt['predict'] = np.concatenate((train_plt_data['predict'], test_plt_data['predict']), axis=0).reshape(-1, )
    # calculate the spread = real - predict
    df_plt['spread'] = np.concatenate((train_plt_data['spread'], test_plt_data['spread']), axis=0).reshape(-1, )
    df_plt['z_score'] = np.concatenate((train_plt_data['z_score'], test_plt_data['z_score']), axis=0).reshape(-1, )
    return df_plt
