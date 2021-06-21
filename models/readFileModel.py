import pandas as pd
import os
from production.codes import config

def read_1min_history_excel(main_path, file_name):
    """
    the timezone is Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments
    :param main_path: str
    :param file_name: str
    :param time_difference_in_hr: time difference between current broker
    :return: pd.DataFrame
    """
    time_difference_in_hr = config.BROKER_TIME_BETWEEN_UTC + config.DOWNLOADED_MIN_DATA_TIME_BETWEEN_UTC
    full_path = os.path.join(main_path, file_name)
    df = pd.read_excel(io=full_path, hearer=None, names=['time', 'open', 'high', 'low', 'close'], usecols='A:E')
    df.set_index('time', inplace=True)
    df.index = df.index.shift(time_difference_in_hr, freq='H')
    return df

def get_file_list(main_path, symbol):
    """
    :param main_path: str
    :param symbol: str
    :return: list
    """
    full_path = os.path.join(main_path, symbol)
    list_dir = os.listdir(full_path)
    sorted(list_dir)
    return list_dir

symbols = ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]

main_path = 'C://Users//Chris//projects//210215_mt5//production//docs//1//min_data'

for symbol in symbols:
    symbol_df = pd.DataFrame()
    list_dir = get_file_list(main_path, symbol)
    for i, file_name in enumerate(list_dir):
        df = read_1min_history_excel(os.path.join(main_path, symbol), file_name)
        if i == 0:
            symbol_df = df.copy()
        else:
            symbol_df = pd.concat([symbol_df, df], axis=0)
    print()
print()