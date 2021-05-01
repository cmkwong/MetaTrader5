import talib

def get_macd(df, fastperiod, slowperiod, signalperiod):
    macd = talib.MACD(df['close'], fastperiod, slowperiod, signalperiod)
    return macd

def get_rsi(df, period):
    rsi = talib.RSI(df['close'], timeperiod=period)
    return rsi

def get_moving_average(df, m_value):
    """
    :param m_value: int
    :return: Series
    """
    moving_average = df['close'].rolling(m_value).sum() / m_value
    return moving_average