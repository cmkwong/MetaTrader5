from production.codes.models import mt5Model, fileModel
from production.codes import config
import os
import numpy as np
import pandas as pd

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
}

file_options = {
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': 'H1',
    'data_time_between_UTC': 5, # that is without daylight shift time (UTC+5)
    'deposit_currency': 'USD',
    'data_path': os.path.join(options['main_path'], "min_data"),
    'output_path': os.path.join(options['main_path'], "ohlc_ret_earning")
}

specific_option = {
    'coefficient_vector': np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389]),
    'lot_times': 10,
}

with mt5Model.Helper():

    symbols_df = None
    for symbol_count, symbol in enumerate(file_options['symbols']):

        print(symbol)
        symbol_df = None
        files_path = os.path.join(file_options['data_path'], symbol)
        min_data_names = fileModel.get_file_list(files_path)

        # concat a symbol in a dataframe (axis = 0)
        for file_count, file_name in enumerate(min_data_names):
            df = fileModel.read_min_history_excel(files_path, file_name, file_options['data_time_between_UTC'])
            if file_count == 0:
                symbol_df = df.copy()
            else:
                symbol_df = pd.concat([symbol_df, df], axis=0)
        symbol_df.drop_duplicates(keep='first', inplace=True) # note 80b

        # concat all symbols in a bigger dataframe (axis = 1)
        if symbol_count == 0:
            symbols_df = symbol_df.copy()
        else:
            symbols_df = pd.concat([symbols_df, symbol_df], axis=1, join='inner')

    # rename columns of the symbols_df
    level_2_arr = np.array(['open', 'high', 'low', 'close'] * len(file_options['symbols']))
    total_list = []
    for symbol in file_options['symbols']:
        total_list = total_list + [symbol] * 4
    level_1_arr = np.array(total_list)
    symbols_df.columns = [level_1_arr, level_2_arr]
    print()