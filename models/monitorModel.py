import numpy as np
import pandas as pd
from production.codes.models.coinModel import nnModel, linearRegressionModel

def get_plotting_data(prices_matrix, model, seq_len):
    """
    :param prices_matrix: accept the train and test prices in array format
    :param model: torch model to get the predicted value
    :param data_options: dict
    :return: array for plotting
    """
    plt_data = {}
    plt_data['inputs'] = prices_matrix[:, :-1]
    plt_data['predict'] = nnModel.get_predicted_arr(plt_data['inputs'], model, seq_len)
    plt_data['target'] = prices_matrix[:, -1]
    return plt_data

def get_plotting_data_simple(prices_matrix, x):
    plt_data = {}
    plt_data['inputs'] = prices_matrix[:, :-1]
    plt_data['predict'] = linearRegressionModel.get_predicted_arr(plt_data['inputs'], x)
    plt_data['target'] = prices_matrix[:, -1]
    return plt_data

def get_plotting_df(train_plt_data, test_plt_data, symbols):
    """
    :param train_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param test_plt_data: data for plotting in dict format: ['inputs','target','predict']
    :param symbols: [str]
    :return: dataframe
    """
    df_plt = pd.DataFrame(columns=symbols)
    for c, symbol in enumerate(symbols[:-1]):
        df_plt[symbol] = np.concatenate((train_plt_data['inputs'][:,c], test_plt_data['inputs'][:,c]), axis=0).reshape(-1, )
    df_plt[symbols[-1]] = np.concatenate((train_plt_data['target'], test_plt_data['target']), axis=0)
    df_plt['predict'] = np.concatenate((train_plt_data['predict'], test_plt_data['predict']), axis=0).reshape(-1, )
    # calculate the spread
    df_plt['spread'] = df_plt[symbols[-1]] - df_plt['predict']
    return df_plt
