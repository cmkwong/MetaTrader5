from production.codes.models import mt5Model, coinModel, timeModel
from production.codes.models.backtestModel import signalModel, priceModel
from production.codes import config
import os
import numpy as np
import time

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
}
trader_options = {
    'symbols': ["AUDJPY","AUDUSD","CADJPY","EURUSD","NZDUSD","USDCAD"],
    'timeframe': '5min', # 1H
    'timezone': "Hongkong",
    'count': 40,
    'deposit_currency': 'USD',
    'history_path': os.path.join(options['main_path'], "history"),
    'max_deviations': [3,3,3,3,3,3],        # the difference between ideal and real price when trading
    'avg_spreads': [18,15,16,16,14,14],     # the max tolerance of spread that accepted
    'type_filling': 'ioc', # ioc / fok / return
    'lot_times': 10
}
coin_option = {
    'coefficient_vector': np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389]),    # will be round to 2 decimal
    'upper_th': 0.001,    # 0.3
    'lower_th': -0.001,   # -0.3
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-100, 500),  # None means no constraint
    'close_change': 1,  # 0 = close; 1 = change
}

with mt5Model.Trader(dt_string=options['dt'], history_path=trader_options["history_path"], type_filling=trader_options['type_filling']) as trader:

    prices_loader = priceModel.Prices_Loader(symbols=trader_options['symbols'],
                                             timeframe=trader_options['timeframe'],
                                             count=trader_options['count'],
                                             timezone=trader_options['timezone'],
                                             deposit_currency=trader_options['deposit_currency'])

    long_lots = [round(i, 2) for i in coinModel.get_modified_coefficient_vector(coin_option['coefficient_vector'], long_mode=True, lot_times=trader_options['lot_times'])]
    short_lots = [round(i, 2) for i in coinModel.get_modified_coefficient_vector(coin_option['coefficient_vector'], long_mode=False, lot_times=trader_options['lot_times'])]

    long_strategy_id, short_strategy_id = coinModel.get_strategy_id(coin_option)
    trader.register_strategy(long_strategy_id, trader_options['symbols'], trader_options['max_deviations'], trader_options['avg_spreads'], trader_options['lot_times'], long_mode=True)
    trader.register_strategy(short_strategy_id, trader_options['symbols'], trader_options['max_deviations'], trader_options['avg_spreads'], trader_options['lot_times'], long_mode=False)

    while True:
        prices_loader.get_data(live=True)
        Prices = prices_loader.Prices
        if not Prices:
            time.sleep(2)
            continue

        # calculate for checking if signal occur
        dependent_variable = Prices.c
        if coin_option['close_change'] == 1:
            dependent_variable = Prices.cc
        coin_data = coinModel.get_coin_data(dependent_variable, coin_option['coefficient_vector'], coin_option['z_score_mean_window'], coin_option['z_score_std_window'])

        # calculate for checking for stop-loss and stop-profit reached
        long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, coin_option['upper_th'], coin_option['lower_th'], discard_head_tail=False)

        trader.strategy_controller(long_strategy_id, Prices.l_o, Prices.l_quote_exchg, coin_option['coefficient_vector'], long_signal, coin_option['slsp'], long_lots)
        trader.strategy_controller(short_strategy_id, Prices.l_o, Prices.l_quote_exchg, coin_option['coefficient_vector'], short_signal, coin_option['slsp'], short_lots)

        time.sleep(5)