import talib

def get_macd(df, fastperiod, slowperiod, signalperiod):
    macd = talib.MACD(df['close'], fastperiod, slowperiod, signalperiod)
    return macd

def get_rsi(df, period):
    rsi = talib.RSI(df['close'], timeperiod=period)
    return rsi

def get_moving_average(close_price, m_value):
    """
    :param close_price: pd.DataFrame
    :param m_value: int
    :return: pd.Series
    """
    moving_average = close_price.rolling(m_value).sum() / m_value
    return moving_average