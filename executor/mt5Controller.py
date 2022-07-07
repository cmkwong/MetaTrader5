import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
from executor import mt5Model
from strategies.Cointegration import coinModel
from data import prices
from backtest import signalModel
import config
import os
import numpy as np
import time

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
    'dt': DT_STRING,
}
trader_options = {
    'symbols': ["AUDJPY","AUDUSD","CADJPY","USDCAD"],
    'timeframe': '1H', # 1H
    'timezone': "Hongkong",
    'count': 40,
    'deposit_currency': 'USD',
    'history_path': os.path.join(options['docs_path'], "history"),
    'max_deviations': [50,50,50,50,50,50],        # the difference between ideal and real price when trading
    'avg_spreads': [50,50,50,50,50,50],     # the max tolerance of spread that accepted - eg: [18,15,16,16,14,14]
    'type_filling': 'ioc', # ioc / fok / return
    'lot_times': 10
}
coin_option = {
    'coefficient_vector': np.array([0.0,0.98467,-0.98578,-0.98662]),    # will be round to 2 decimal
    'upper_th': 1.5, # 1.5
    'lower_th': -1.5, # -1.5
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-50000, 50000),  # None means no constraint
    'close_change': 1,  # 0 = close; 1 = change
}

with mt5Model.Trader(dt_string=options['dt'], history_path=trader_options["history_path"], type_filling=trader_options['type_filling']) as trader:

    prices_loader = prices.Prices_Loader(symbols=trader_options['symbols'],
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
        prices_loader.get_data(latest=True)
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

        trader.strategy_controller(long_strategy_id, Prices.c, Prices.quote_exchg, coin_option['coefficient_vector'], long_signal, coin_option['slsp'], long_lots)
        trader.strategy_controller(short_strategy_id, Prices.c, Prices.quote_exchg, coin_option['coefficient_vector'], short_signal, coin_option['slsp'], short_lots)

        time.sleep(5)