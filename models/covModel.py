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
            price = mt5Model.get_historical_data(start, end, symbol, timeframe, timezone)
            price = price.set_index('time')['close']
            if i == 0:
                price_matrix = price
            else:
                price_matrix = pd.concat([price_matrix, price], axis=1, join='inner')
    return price_matrix.values

def corela_table(cor_matrix, symbol_list):
    cor_table = pd.DataFrame(cor_matrix, index=symbol_list, columns=symbol_list)
    return cor_table