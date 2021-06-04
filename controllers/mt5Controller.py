from production.codes.models import mt5Model, priceModel, coinModel
from production.codes.models.backtestModel import signalModel
from production.codes import config
import numpy as np
import time

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION),
    'dt': DT_STRING,
    'debug': True,
    'lot_times': 100
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'count': 40,
    'deposit_currency': 'USD',
    'price_plt_save_path': options['main_path'] + "coin_plt/",
}

coin_option = {
    'upper_th': 0.3,
    'lower_th': -0.3,
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-100, 5000),  # None means no constraint
}

with mt5Model.Trader(deviation=5, type_filling='ioc') as trader:

    coefficient_vector = np.array([2.58766,0.01589,-1.76342,-0.01522,0.00351,0.01389])
    long_lots = [round(i * options['lot_times'],2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=True)]
    short_lots = [round(i * options['lot_times'],2) for i in coinModel.get_modify_coefficient_vector(coefficient_vector, long_mode=False)]

    long_strategy_id, short_strategy_id = coinModel.get_strategy_id(coin_option)
    trader.register_strategy(long_strategy_id, data_options['symbols'], long_lots)
    trader.register_strategy(short_strategy_id, data_options['symbols'], short_lots)

    while True:
        Prices = priceModel.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                       start=None, end=None, ohlc='1111', count=data_options['count'], deposit_currency=data_options['deposit_currency'])

        # calculate for checking if signal occur
        coin_data = coinModel.get_coin_data(Prices.c, coefficient_vector, coin_option['z_score_mean_window'], coin_option['z_score_std_window'])

        # calculate for checking for stop-loss and stop-profit reached
        long_signal, short_signal = signalModel.get_coin_NN_signal(coin_data, coin_option['upper_th'], coin_option['lower_th'], discard=False)
        masked_open_prices = priceModel.get_open_price_masked_with_last_price(Prices.o, Prices.c)  # masked with last price with close price

        # # lots for long / short
        # long_lots = list(get_modify_coefficient_vector(coefficient_vector, long_mode=True))
        # short_lots = list(get_modify_coefficient_vector(coefficient_vector, long_mode=False))

        coinModel.get_action(trader, long_strategy_id, masked_open_prices, Prices.quote_exchg, Prices.ptDv, coefficient_vector, long_signal, coin_option['slsp'], long_mode=True)
        coinModel.get_action(trader, short_strategy_id, masked_open_prices, Prices.quote_exchg, Prices.ptDv, coefficient_vector, short_signal, coin_option['slsp'], long_mode=False)

        time.sleep(5)


#     lots = [-1.59,176.43,1.52,-0.42,-1.45, 100]
#     symbols = ['AUDJPY','AUDUSD','CADJPY','EURUSD','NZDUSD','USDCAD']
#     requests = mt5Model.requests_format(symbols, lots, deviation=5, type_filling='fok')
#     order_ids = mt5Model.requests_execute(requests)
#