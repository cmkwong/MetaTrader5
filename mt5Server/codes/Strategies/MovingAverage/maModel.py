import numpy as np
import pandas as pd

from mt5Server.codes.Views import plotPre

from myBacktest import techModel, signalModel
from myUtils import printModel, dicModel

def get_optimize_moving_average_csv_text(Prices, limit_unit, max_index=201):
    long_stat_csv_txt, short_stat_csv_txt = '', ''
    long_stat, short_stat = {}, {}
    for slow_index in range(1, max_index):
        for fast_index in range(1, slow_index):
            if slow_index == fast_index:
                continue
            # moving average object
            long_ma_data = get_ma_data(Prices.c, fast_index, slow_index)
            short_ma_data = get_ma_data(Prices.c, fast_index, slow_index)
            long_signal, short_signal = signalModel.get_movingAverage_signal(long_ma_data, short_ma_data, limit_unit=limit_unit)

            Graph_Data = plotPre._get_graph_data(Prices, long_signal, short_signal, coefficient_vector=np.array([]))

            # stat for both long and short (including header)
            long_stat = Graph_Data.stats['long']['earning']
            short_stat = Graph_Data.stats['short']['earning']
            long_stat['limit_unit'], long_stat['slow'], long_stat['fast'] = limit_unit, slow_index, fast_index
            short_stat['limit_unit'], short_stat['slow'], short_stat['fast'] = limit_unit, slow_index, fast_index

            if long_stat["total"] > 0: long_stat_csv_txt = dicModel.append_dictValues_into_text(long_stat, long_stat_csv_txt)
            if short_stat["total"] > 0: short_stat_csv_txt = dicModel.append_dictValues_into_text(short_stat, short_stat_csv_txt)

            # # print results
            print("\nlimit unit: {}; slow index: {}; fast index: {}".format(limit_unit, slow_index, fast_index))
            printModel.print_dict(long_stat)
            printModel.print_dict(short_stat)
    # added to header to the text
    long_stat_csv_txt = ','.join(list(long_stat.keys())) + '\n' + long_stat_csv_txt
    short_stat_csv_txt = ','.join(list(short_stat.keys())) + '\n' + short_stat_csv_txt
    return long_stat_csv_txt, short_stat_csv_txt

def get_ma_data(close_prices, fast_param, slow_param):
    ma_data = pd.DataFrame(index=close_prices.index)
    ma_data['fast'] = techModel.get_moving_average(close_prices, fast_param)
    ma_data['slow'] = techModel.get_moving_average(close_prices, slow_param)
    return ma_data

