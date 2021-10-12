import talib
import pandas as pd
import numpy as np

def get_macd(closes, fastperiod, slowperiod, signalperiod):
    # define 2 level of columns for dataframe
    symbols = closes.columns
    level_2_arr = np.array(['value', 'signal', 'hist'] * len(symbols))
    level_1_arr = np.array([[symbol] * 3 for symbol in symbols])
    column_index_arr = [
        level_1_arr, level_2_arr
    ]
    macd = pd.DataFrame(columns=column_index_arr, index=closes.index)
    # calculate and assign the value
    for symbol in closes.columns:
        macd[symbol]['value'], macd[symbol]['signal'], macd[symbol]['hist'] = talib.MACD(closes[symbol], fastperiod, slowperiod, signalperiod)
    return macd

def get_rsi(closes, period):
    rsi = pd.DataFrame(columns=closes.columns, index=closes.index)
    for symbol in closes.columns:
        rsi[symbol] = talib.RSI(closes[symbol], timeperiod=period)
    return rsi

def get_moving_average(closes, m_value):
    """
    :param closes: pd.DataFrame
    :param m_value: int
    :return: pd.DataFrame
    """
    ma = pd.DataFrame(columns=closes.columns, index=closes.index)
    for symbol in closes.columns:
        ma[symbol] = closes[symbol].rolling(m_value).sum() / m_value
    return ma

def get_bollinger_band(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    :param closes: pd.DataFrame
    :param timeperiod: int
    :param nbdevup: int
    :param nbdevdn: int
    :param matype: int, #MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: upperband (pd.DataFrame), middleband (pd.DataFrame), lowerband (pd.DataFrame)
    """
    # define 2 level of columns for dataframe
    symbols = closes.columns
    level_2_arr = np.array(['upper', 'middle', 'lower'] * len(symbols))
    level_1_arr = np.array([[symbol] * 3 for symbol in symbols])
    column_index_arr = [
        level_1_arr, level_2_arr
    ]
    bb = pd.DataFrame(columns=column_index_arr, index=closes.index)
    # calculate and assign the value
    for symbol in closes.columns:
        bb[symbol]['upper'], bb[symbol]['middle'], bb[symbol]['lower'] = talib.BBANDS(closes[symbol], timeperiod, nbdevup, nbdevdn, matype)
    return bb

def get_stochastic_oscillator(highs, lows, closes, fastk_period=5, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
    """
    :param highs: pd.DataFrame
    :param lows: pd.DataFrame
    :param closes: pd.DataFrame
    :param fastk_period: int
    :param slowk_period: int
    :param slowk_matype: int, MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :param slowd_period: int
    :param slowd_matype: int, MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: slowk (pd.DataFrame), slowd (pd.DataFrame)
    """
    # define 2 level of columns for dataframe
    symbols = closes.columns
    level_2_arr = np.array(['k', 'd'] * len(symbols))
    level_1_arr = np.array([[symbol] * 2 for symbol in symbols])
    column_index_arr = [
        level_1_arr, level_2_arr
    ]
    stocOsci = pd.DataFrame(columns=column_index_arr, index=closes.index)
    # calculate and assign the value
    for symbol in closes.columns:
        stocOsci[symbol]['k'], stocOsci[symbol]['d'] = talib.STOCH(highs[symbol], lows[symbol], closes[symbol], fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    return stocOsci

def get_standard_deviation(closes, timeperiod, nbdev=0):
    """
    :param closes: pd.DataFrame
    :param timeperiod: int
    :param nbdev: int,
    :return: pd.DataFrame
    """
    std = pd.DataFrame(columns=closes.columns, index=closes.index)
    for symbol in closes.columns:
        std[symbol] = talib.STDDEV(closes[symbol], timeperiod, nbdev)
    return std

def get_tech_datas(Prices, params, tech_name):
    """
    :param Prices: collection object
    :param params: {'ma': [param]}
    :param tech_name: str
    :return:
    """
    datas = {}
    for param in params:
        if tech_name == 'ma':
            datas[param] = get_moving_average(Prices.c, param)
        elif tech_name == 'bb':
            datas[param] = get_bollinger_band(Prices.c, *param)
        elif tech_name == 'std':
            datas[param] = get_standard_deviation(Prices.c, *param)
        elif tech_name == 'rsi':
            datas[param] = get_rsi(Prices.c, param)
        elif tech_name == 'stocOsci':
            datas[param] = get_stochastic_oscillator(Prices.h, Prices.l, Prices.c, *param)
        elif tech_name == 'macd':
            datas[param] = get_macd(Prices.c, *param)
    return datas
