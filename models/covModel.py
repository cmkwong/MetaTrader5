from production.codes.models import mt5Model
from production.codes.controllers import mt5Controller
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
    raw_prices = []
    with mt5Controller.Helper():
        for i, symbol in enumerate(symbols):
            print(symbol)
            raw_price = mt5Model.get_historical_data(start, end, symbol, timeframe, timezone)
            price = raw_price['close'].to_numpy().reshape(-1,1)
            raw_prices.append(raw_price)
            if i == 0:
                price_matrix = price
            else:
                price_matrix = np.concatenate((price_matrix, price), axis=1)

    return price_matrix

def z_col(col):
    mean = np.mean(col)
    std = np.std(col)
    normalized_col = (col - mean) / std
    return normalized_col

from production.codes import config
price_matrix = prices_matrix(config.START, config.END, ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURCAD","USDCAD", "AUDUSD", "EURGBP", "NZDUSD"], config.TIMEFRAME, config.TIMEZONE)
cor = corela_matrix(price_matrix)
print()