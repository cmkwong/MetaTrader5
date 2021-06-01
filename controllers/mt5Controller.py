from production.codes.models import mt5Model, priceModel
from production.codes import config
from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION),
    'dt': DT_STRING,
    'debug': True
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'count': 20,
    'deposit_currency': 'USD',
    'price_plt_save_path': options['main_path'] + "coin_plt/",
}

with mt5Model.Helper():
    Prices = priceModel.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                          start=None, end=None, ohlc='1111', count=data_options['count'], deposit_currency=data_options['deposit_currency'])

#     lots = [-1.59,176.43,1.52,-0.42,-1.45, 100]
#     symbols = ['AUDJPY','AUDUSD','CADJPY','EURUSD','NZDUSD','USDCAD']
#     requests = mt5Model.requests_format(symbols, lots, deviation=5, type_filling='fok')
#     order_ids = mt5Model.requests_execute(requests)
#
    print()