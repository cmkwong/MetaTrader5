from production.codes.models import mt5Model
from production.codes.models import covModel

data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD",
               "EURNOK", "EURSEK", "AUDJPY", "EURSGD", "GBPSGD", "GBPAUD", "CADJPY"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'shuffle': True,
    'trainTestSplit': 0.7,
    'seq_len': 20,
    'batch_size': 32,
}
# config.START, config.END, symbol_list, config.TIMEFRAME, config.TIMEZONE
def get_cor_matrix(start, end, symbol_list, timeframe, timezone):
    symbol_list = sorted(symbol_list, reverse=False)# sorting the symbol_list
    print(symbol_list)
    price_matrix = mt5Model.get_prices_matrix(symbol_list, timeframe, timezone, start, end)
    cor_matrix = covModel.corela_matrix(price_matrix)
    cor_table = covModel.corela_table(cor_matrix, symbol_list)
    return cor_matrix, cor_table

cor_matrix, cor_table = get_cor_matrix(data_options['start'], data_options['end'], data_options['symbols'], data_options['timeframe'], data_options['timezone'])
print()