from production.codes import config
from production.codes.models import mt5Model
from production.codes.models.backtestModel import priceModel
from production.codes.models import covModel
import os

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'debug': True,
    'local': False
}

data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD",
                "AUDJPY", "GBPAUD", "CADJPY"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'local_min_path': os.path.join(options['main_path'], "min_data"),
}
# config.START, config.END, symbols, config.TIMEFRAME, config.TIMEZONE
def get_cor_matrix(prices_loader, local):
    """
    :param prices_loader: class object: Prices_Loader
    :param local: Boolean
    :return:
    """
    Prices = prices_loader.format_Prices(local)
    price_matrix = Prices.cc.values # note 83i
    cor_matrix = covModel.corela_matrix(price_matrix)
    cor_table = covModel.corela_table(cor_matrix, symbols)
    return cor_matrix, cor_table

with mt5Model.Helper():
    symbols = sorted(data_options['symbols'], reverse=False)  # sorting the symbols
    prices_loader = priceModel.Prices_Loader(symbols=symbols,
                                             timeframe=data_options['timeframe'],
                                             start=data_options['start'],
                                             end=data_options['end'],
                                             timezone=data_options['timezone'],
                                             data_path=data_options['local_min_path'],
                                             deposit_currency=data_options['deposit_currency'])

    cor_matrix, cor_table = get_cor_matrix(prices_loader, options['local'])
print()