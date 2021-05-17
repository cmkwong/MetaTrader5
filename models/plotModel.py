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



