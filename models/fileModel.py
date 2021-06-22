import pandas as pd
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
    df = pd.read_excel(io=full_path, hearer=None, names=['time', 'open', 'high', 'low', 'close'], usecols='A:E')
    df.set_index('time', inplace=True)
    df.index = df.index.shift(time_difference_in_hr, freq='H')
    return df

def get_file_list(files_path):
    """
    :param files_path: str, data_path + symbol
    :param symbol: str
    :return: list
    """
    list_dir = os.listdir(files_path)
    sorted(list_dir)
    return list_dir