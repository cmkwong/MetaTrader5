import pandas as pd
import numpy as np
import os
from production.codes import config

def read_min_history_excel(main_path, file_name, data_time_difference_to_UTC):
    """
    the timezone is Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments
    :param main_path: str
    :param file_name: str
    :param time_difference_in_hr: time difference between current broker
    :return: pd.DataFrame
    """
    time_difference_in_hr = config.BROKER_TIME_BETWEEN_UTC + data_time_difference_to_UTC
    full_path = os.path.join(main_path, file_name)
    df = pd.read_excel(io=full_path, hearer=None, names=['time', 'open'], usecols='A:B')
    df.set_index('time', inplace=True)
    df.index = df.index.shift(time_difference_in_hr, freq='H')
    return df

def get_file_list(files_path, reverse=False):
    """
    :param files_path: str, data_path + symbol
    :param symbol: str
    :return: list
    """
    list_dir = os.listdir(files_path)
    list_dir = sorted(list_dir, reverse=reverse)
    return list_dir

def clear_files(main_path):
    # clear before write
    for file in get_file_list(main_path):
        remove_full_path = os.path.join(main_path, file)
        os.remove(remove_full_path)
        print("The file {} has been removed.".format(file))

def write_min_extra_info(main_path, file_name, symbols, long_signal, short_signal, long_modify_exchg_q2d, short_modify_exchg_q2d):
    """
    :param main_path: str
    :param file_name: str
    :param symbols: list
    :param long_signal: pd.Series
    :param short_signal: pd.Series
    :param long_modify_exchg_q2d: pd.DataFrame
    :param short_modify_exchg_q2d: pd.DataFrame
    :return: None
    """
    # concat the data axis=1
    df_for_min = pd.concat([long_signal, short_signal, long_modify_exchg_q2d, short_modify_exchg_q2d], axis=1)
    # re-name
    level_2_arr = np.array(['long', 'short'] + symbols * 2)
    level_1_arr = np.array(['signal'] * 2 + ['long_q2d'] * len(symbols) + ['short_q2d'] * len(symbols))
    df_for_min.columns = [level_1_arr, level_2_arr]
    df_for_min.to_csv(os.path.join(main_path, file_name))
    print("Extra info write to {}".format(main_path))

def read_min_extra_info(main_path):
    """
    :param main_path: str
    :param col_list: list, [str/int]: required column names
    :return: Series, Series, DataFrame, DataFrame
    """
    file_names = get_file_list(main_path, reverse=True)
    dfs = None
    for i, file_name in enumerate(file_names):
        full_path = os.path.join(main_path, file_name)
        df = pd.read_csv(full_path, header=[0, 1], index_col=0)
        if i == 0 :
            dfs = df.copy()
        else:
            dfs = pd.concat([dfs, df], axis=0)
    # di-assemble into different parts
    long_signal = dfs.loc[:, ('signal', 'long')]
    short_signal = dfs.loc[:, ('signal', 'short')]
    long_q2d = dfs.loc[:, ('long_q2d')]
    short_q2d = dfs.loc[:, ('short_q2d')]
    return long_signal, short_signal, long_q2d, short_q2d