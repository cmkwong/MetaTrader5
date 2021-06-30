from production.codes.models import mt5Model, fileModel
from production.codes.models.backtestModel import signalModel, exchgModel, pointsModel, returnModel, priceModel
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

    symbols_min_prices = priceModel._get_local_prices(file_options['data_path'], file_options['symbols'], file_options['data_time_difference_to_UTC'], file_options['timeframe'], '1001')

    # make the extra data in higher resolution
    print("Processing: Resolution of the extra data")
    resoluted_index = symbols_min_prices.index
    resoluted_long_signal = signalModel.get_resoluted_signal(long_signal, resoluted_index)
    resoluted_short_signal = signalModel.get_resoluted_signal(short_signal, resoluted_index)
    resoluted_modified_long_q2d = exchgModel.get_resoluted_exchg(long_modified_q2d, long_signal, resoluted_index)
    resoluted_modified_short_q2d = exchgModel.get_resoluted_exchg(short_modified_q2d, short_signal, resoluted_index)
    
    # get points_dff_values_df
    all_symbols_info = mt5Model.get_all_symbols_info()
    points_dff_values_df = pointsModel.get_points_dff_values_df(file_options['symbols'], symbols_min_prices, symbols_min_prices.shift(1), all_symbols_info)
    
    # get ret and earning
    long_ret, long_earning = returnModel.get_ret_earning(symbols_min_prices, symbols_min_prices.shift(1), resoluted_modified_long_q2d, points_dff_values_df,
                                                         specific_option['coefficient_vector'], long_mode=True, lot_times=specific_option['lot_times'],
                                                         shift_offset=(60, 'min'))
    short_ret, short_earning = returnModel.get_ret_earning(symbols_min_prices, symbols_min_prices.shift(1), resoluted_modified_short_q2d, points_dff_values_df,
                                                           specific_option['coefficient_vector'], long_mode=False, lot_times=specific_option['lot_times'],
                                                           shift_offset=(60, 'min'))

    # ret_earning = pd.concat([long_ret, short_ret, long_earning, short_earning], axis=1)

    # write the csv
    # csv_name = "{}_min_ret_earning.csv".format(DT_STRING)
    # print("Writing csv to {}".format(file_options['output_path']))
    # ret_earning.to_csv(os.path.join(file_options['output_path'], csv_name))

    # debug
    debug_df = pd.concat([symbols_min_prices, resoluted_long_signal, resoluted_short_signal,
                          resoluted_modified_long_q2d, resoluted_modified_short_q2d,
                          points_dff_values_df,
                          long_ret, short_ret, long_earning, short_earning], axis=1)
    csv_name = "{}_debug.csv".format(DT_STRING)
    print("Writing debug csv to {}".format(file_options['output_path']))
    debug_df.head(100000).to_csv(os.path.join(file_options['output_path'], 'head_' + csv_name))
    debug_df.tail(100000).to_csv(os.path.join(file_options['output_path'], 'tail_' + csv_name))
    print()