from production.codes.models import mt5Model, priceModel, coinModel, timeModel
from production.codes.models.backtestModel import signalModel
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
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': timeModel.get_txt2timeframe('M5'), # H1
    'timezone': "Hongkong",
    'count': 40,
    'deposit_currency': 'USD',
    'history_path': os.path.join(options['main_path'], "history"),
    'max_deviations': [3,3,3,3,3,3],
    'avg_spreads': [18,15,16,16,14,14],
    'type_filling': 'ioc', # ioc / fok / return
    'lot_times': 10
}
coin_option = {
    'upper_th': 0.001,    # 0.3
    'lower_th': -0.001,   # -0.3
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-100, 5000),  # None means no constraint
}

with mt5Model.Trader(dt_string=options['dt'], history_path=trader_options["history_path"], type_filling=trader_options['type_filling']) as trader:

    coefficient_vector = np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389]) # will be round to 2 decimal
    long_lots = [round(i, 2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=True, lot_times=trader_options['lot_times'])]
    short_lots = [round(i, 2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=False, lot_times=trader_options['lot_times'])]

    long_strategy_id, short_strategy_id = coinModel.get_strategy_id(coin_option)
    trader.register_strategy(long_strategy_id, trader_options['symbols'], trader_options['max_deviations'], trader_options['avg_spreads'], trader_options['lot_times'])
    trader.register_strategy(short_strategy_id, trader_options['symbols'], trader_options['max_deviations'], trader_options['avg_spreads'], trader_options['lot_times'])

    while True:
        Prices = priceModel.get_latest_Prices(trader.all_symbol_info, trader_options['symbols'], trader_options['timeframe'], trader_options['timezone'],
                                       count=trader_options['count'], deposit_currency=trader_options['deposit_currency'])
        if not Prices:
            time.sleep(2)
            continue

        # calculate for checking if signal occur
        coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector, coin_option['z_score_mean_window'], coin_option['z_score_std_window'])

        # calculate for checking for stop-loss and stop-profit reached
        long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, coin_option['upper_th'], coin_option['lower_th'], discard=False)

        coinModel.get_action(trader, long_strategy_id, Prices.l_o, Prices.l_quote_exchg,
                             coefficient_vector, long_signal, coin_option['slsp'], long_lots, long_mode=True)
        coinModel.get_action(trader, short_strategy_id, Prices.l_o, Prices.l_quote_exchg,
                             coefficient_vector, short_signal, coin_option['slsp'], short_lots, long_mode=False)

        time.sleep(5)