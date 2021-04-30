from production.codes.models import mt5Model
from production.codes.controllers import mt5Controller
import pandas as pd
import numpy as np

def cov_matrix(array_2d, rowvar=False, bias=False):
    matrix = np.cov(array_2d, rowvar=rowvar, bias=bias)
    return matrix

def corela_matrix(array_2d, rowvar=False, bias=False):
    matrix = np.corrcoef(array_2d, rowvar=rowvar, bias=bias)
    return matrix

def prices_matrix(start, end, symbols, timeframe, timezone):
    """
    :param start: (2010,1,1,0,0)
    :param end:  (2020,1,1,0,0)
    :param symbols: [str]
    :param timeframe: config.TIMEFRAME
    :param timezone: str "Etc/UTC"
    :return:
    """
    price_matrix = None
    with mt5Controller.Helper():
        for i, symbol in enumerate(symbols):
            print(symbol)
            price = mt5Model.get_historical_data(start, end, symbol, timeframe, timezone)['close']
            if i == 0:
                price_matrix = price
            else:
                price_matrix = pd.concat([price_matrix, price], axis=1, join='inner')
    return price_matrix.values

def z_col(col):
    mean = np.mean(col)
    std = np.std(col)
    normalized_col = (col - mean) / std
    return normalized_col

from production.codes import config
symbol_list = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD"]
price_matrix = prices_matrix(config.START, config.END, symbol_list, config.TIMEFRAME, config.TIMEZONE)
cor_matrix = corela_matrix(price_matrix)
cor_table = pd.DataFrame(cor_matrix, index=symbol_list, columns=symbol_list)
print()