from production.codes.models import mt5Model, timeModel
from production.codes.models.backtestModel import priceModel
from production.codes.models import covModel

data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD",
               "EURNOK", "EURSEK", "AUDJPY", "EURSGD", "GBPSGD", "GBPAUD", "CADJPY"],
    'timeframe': timeModel.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
}
# config.START, config.END, symbols, config.TIMEFRAME, config.TIMEZONE
def get_cor_matrix(symbols, start, end, timeframe, timezone, deposit_currency):
    symbols = sorted(symbols, reverse=False)# sorting the symbols
    print(symbols)
    Prices = priceModel.get_Prices(symbols, timeframe, timezone, start, end=end, ohlc='1111', deposit_currency=deposit_currency)
    price_matrix = Prices.c.values
    cor_matrix = covModel.corela_matrix(price_matrix)
    cor_table = covModel.corela_table(cor_matrix, symbols)
    return cor_matrix, cor_table

with mt5Model.Helper():
    cor_matrix, cor_table = get_cor_matrix(data_options['symbols'], data_options['start'], data_options['end'],
                                       data_options['timeframe'], data_options['timezone'], data_options['deposit_currency'])
print()