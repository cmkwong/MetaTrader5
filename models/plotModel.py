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

def get_plotting_data(prices_df, model, seq_len):
    """
    :param prices_matrix: accept the train and test prices in array format
    :param model: torch model to get the predicted value
    :param data_options: dict
    :return: array for plotting
    """
    plt_df = pd.DataFrame(index=prices_df.index)
    for symbol in prices_df.columns:
        plt_df[symbol] = prices_df[symbol]
    plt_df['predict'] = model.get_predicted_arr(prices_df.iloc[:,:-1].values, seq_len)
    spread = prices_df.iloc[:, -1] - plt_df['predict']
    plt_df['spread'] = spread
    plt_df['z_score'] = maths.z_score_with_rolling_mean(spread.values, 10)
    return plt_df

def get_plotting_data_simple(prices_df, coefficient_vector):
    """
    :param prices_df: accept the train and test prices in pd.dataframe format
    :param coefficient_vector:
    :return:
    """
    plt_df = pd.DataFrame(index=prices_df.index)
    for symbol in prices_df.columns:
        plt_df[symbol] = prices_df[symbol]
    plt_df['predict'] = coinModel.get_predicted_arr(prices_df.iloc[:,:-1].values, coefficient_vector)
    spread = prices_df.iloc[:,-1] - plt_df['predict']
    plt_df['spread'] = spread
    plt_df['z_score'] = maths.z_score_with_rolling_mean(spread.values, 10)
    return plt_df

def concatenate_plotting_df(train_plt_df, test_plt_df):
    """
    :param train_plt_df: pd.Dataframe
    :param test_plt_df: pd.Dataframe
    return df_plt: pd.Dataframe
    """
    df_plt = pd.concat(train_plt_df,test_plt_df)
    return df_plt

def concatenate_plotting_df_b(train_plt_data, test_plt_data, time_index, symbols):
    """
    :param train_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param test_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param time_index: timeframe
    :param symbols: [str]
    :return: dataframe
    """
    target = symbols[-1]
    df_plt = pd.DataFrame(index=time_index, columns=symbols)
    for c, symbol in enumerate(symbols[:-1]):
        df_plt[symbol] = np.concatenate((train_plt_data['inputs'][:,c], test_plt_data['inputs'][:,c]), axis=0).reshape(-1, )
    df_plt[target] = np.concatenate((train_plt_data['target'], test_plt_data['target']), axis=0)
    df_plt['predict'] = np.concatenate((train_plt_data['predict'], test_plt_data['predict']), axis=0).reshape(-1, )
    # calculate the spread = real - predict
    df_plt['spread'] = np.concatenate((train_plt_data['spread'], test_plt_data['spread']), axis=0).reshape(-1, )
    df_plt['z_score'] = np.concatenate((train_plt_data['z_score'], test_plt_data['z_score']), axis=0).reshape(-1, )
    return df_plt
