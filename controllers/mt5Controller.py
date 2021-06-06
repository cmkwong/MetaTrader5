from production.codes.models import mt5Model, priceModel, coinModel
from production.codes.models.backtestModel import signalModel
from production.codes import config
import os
import numpy as np
import time

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION),
    'dt': DT_STRING,
}
trader_options = {
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'count': 40,
    'deposit_currency': 'USD',
    'history_path': os.path.join(options['main_path'], "history"),
    'deviations': [6, 4, 7, 5, 8, 5],
    'type_filling': 'ioc', # ioc / fok / return
    'lot_times': 100
}
coin_option = {
    'upper_th': 0.3,
    'lower_th': -0.3,
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-100, 5000),  # None means no constraint
}

with mt5Model.Trader(history_path=trader_options["history_path"], deviations=trader_options['deviations'], type_filling=trader_options['type_filling']) as trader:

    coefficient_vector = np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389]) # will be round to 2 decimal
    long_lots = [round(i * trader_options['lot_times'], 2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=True)]
    short_lots = [round(i * trader_options['lot_times'], 2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=False)]

    long_strategy_id, short_strategy_id = coinModel.get_strategy_id(coin_option)
    trader.register_strategy(long_strategy_id, trader_options['symbols'])
    trader.register_strategy(short_strategy_id, trader_options['symbols'])

    while True:
        Prices = priceModel.get_Prices(trader_options['symbols'], trader_options['timeframe'], trader_options['timezone'],
                                       start=None, end=None, ohlc='1111', count=trader_options['count'], deposit_currency=trader_options['deposit_currency'])

        # calculate for checking if signal occur
        coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector, coin_option['z_score_mean_window'], coin_option['z_score_std_window'])

        # calculate for checking for stop-loss and stop-profit reached
        long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, coin_option['upper_th'], coin_option['lower_th'], discard=False)
        masked_open_prices = priceModel.get_open_price_masked_with_last_price(Prices.o, Prices.c)  # masked with last price with close price

        coinModel.get_action(trader, long_strategy_id, masked_open_prices, Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_signal, coin_option['slsp'], long_lots, long_mode=True)
        coinModel.get_action(trader, short_strategy_id, masked_open_prices, Prices.quote_exchg, Prices.ptDv, coefficient_vector, short_signal, coin_option['slsp'], short_lots, long_mode=False)

        time.sleep(30)
