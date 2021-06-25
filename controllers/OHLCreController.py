from production.codes.models import mt5Model, fileModel
from production.codes.models.backtestModel import signalModel, exchgModel, pointsModel, returnModel
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
    'extra_data_path': os.path.join(options['main_path'], 'min_data\extra_data'),
    'output_path': os.path.join(options['main_path'], "ohlc_ret_earning")
}
specific_option = {
    'coefficient_vector': np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389]),
    'lot_times': 10,
}

with mt5Model.Helper():

    # read extra data
    long_signal, short_signal, long_modified_q2d, short_modified_q2d = fileModel.read_min_extra_info(file_options['extra_data_path'])

    for symbol_count, symbol in enumerate(file_options['symbols']):
        print("Processing: {}".format(symbol))
        files_path = os.path.join(file_options['data_path'], symbol)
        min_data_names = fileModel.get_file_list(files_path)

        # concat a symbol in a dataframe (axis = 0)
        for file_count, file_name in enumerate(min_data_names):
            df = fileModel.read_min_history_excel(files_path, file_name, file_options['data_time_between_UTC'])
            if file_count == 0:
                symbol_prices = df.copy()
            else:
                symbol_prices = pd.concat([symbol_prices, df], axis=0)
        # drop the duplicated index row
        symbol_prices = symbol_prices[~symbol_prices.index.duplicated(keep='first')] # note 80b and note 81c

        # join='outer' method with all symbols in a bigger dataframe (axis = 1)
        if symbol_count == 0:
            symbols_prices = symbol_prices.copy()
        else:
            symbols_prices = pd.concat([symbols_prices, symbol_prices], axis=1, join='outer')

    # replace NaN values with preceding values
    symbols_prices.fillna(method='ffill', inplace=True)

    # rename columns of the symbols_prices
    level_2_arr = np.array(['open'] * len(file_options['symbols']))
    level_1_arr = np.array(file_options['symbols'])
    symbols_prices.columns = [level_1_arr, level_2_arr]

    # make the extra data in higher resolution
    print("Processing: Resolution of the extra data")
    resoluted_index = symbols_prices.index
    resoluted_long_signal = signalModel.get_resoluted_signal(long_signal, resoluted_index)
    resoluted_short_signal = signalModel.get_resoluted_signal(short_signal, resoluted_index)
    resoluted_modified_long_q2d = exchgModel.get_resoluted_exchg(long_modified_q2d, long_signal, resoluted_index)
    resoluted_modified_short_q2d = exchgModel.get_resoluted_exchg(short_modified_q2d, short_signal, resoluted_index)
    
    # get points_dff_values_df
    all_symbols_info = mt5Model.get_all_symbols_info()
    points_dff_values_df = pointsModel.get_points_dff_values_df(file_options['symbols'], symbols_prices, symbols_prices.shift(1), all_symbols_info)
    
    # get ret and earning
    long_ret, long_earning = returnModel.get_ret_earning(symbols_prices, symbols_prices.shift(1), resoluted_modified_long_q2d, points_dff_values_df,
                                                         specific_option['coefficient_vector'], long_mode=True, lot_times=specific_option['lot_times'],
                                                         shift_offset=(60, 'min'))
    short_ret, short_earning = returnModel.get_ret_earning(symbols_prices, symbols_prices.shift(1), resoluted_modified_short_q2d, points_dff_values_df,
                                                           specific_option['coefficient_vector'], long_mode=False, lot_times=specific_option['lot_times'],
                                                           shift_offset=(60, 'min'))

    # ret_earning = pd.concat([long_ret, short_ret, long_earning, short_earning], axis=1)

    # write the csv
    # csv_name = "{}_min_ret_earning.csv".format(DT_STRING)
    # print("Writing csv to {}".format(file_options['output_path']))
    # ret_earning.to_csv(os.path.join(file_options['output_path'], csv_name))

    # debug
    debug_df = pd.concat([symbols_prices, resoluted_long_signal, resoluted_short_signal,
                          resoluted_modified_long_q2d, resoluted_modified_short_q2d,
                          points_dff_values_df,
                          long_ret, short_ret, long_earning, short_earning], axis=1)
    csv_name = "{}_debug.csv".format(DT_STRING)
    print("Writing debug csv to {}".format(file_options['output_path']))
    debug_df.head(100000).to_csv(os.path.join(file_options['output_path'], 'head_' + csv_name))
    debug_df.tail(100000).to_csv(os.path.join(file_options['output_path'], 'tail_' + csv_name))
    print()