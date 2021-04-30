from production.codes.models import covModel
from production.codes import config

symbol_list = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD",
               "EURNOK", "EURSEK", "AUDJPY", "EURSGD", "GBPSGD", "GBPAUD", "CADJPY"]

# config.START, config.END, symbol_list, config.TIMEFRAME, config.TIMEZONE
def get_cor_matrix(start, end, symbol_list, timeframe, timezone):
    symbol_list = sorted(symbol_list, reverse=False)# sorting the symbol_list
    print(symbol_list)
    price_matrix = covModel.prices_matrix(start, end, symbol_list, timeframe, timezone)
    cor_matrix = covModel.corela_matrix(price_matrix)
    cor_table = covModel.corela_table(cor_matrix, symbol_list)
    return cor_matrix, cor_table

cor_matrix, cor_table = get_cor_matrix(config.START, config.END, symbol_list, config.TIMEFRAME, config.TIMEZONE)
print()