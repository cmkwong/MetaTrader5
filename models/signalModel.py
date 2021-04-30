from production.codes.models import indexModel, techModel

def discard_head_signal(signal):
    """
    :param signal: Series
    :return: signal: Series
    """
    if signal[0] == True:
        for index, value in signal.items():
            if value == True:
                signal[index] = False
            else:
                break
    return signal

def discard_tail_signal(signal):
    """
    :param signal: Series
    :return: signal: Series
    """
    if signal[len(signal) - 1] == True or signal[len(signal) - 2] == True:  # See Note 6. and 11.
        length = len(signal)
        signal[length - 1] = True  # Set the last index is True, it will set back to false in following looping
        for ii, value in enumerate(reversed(signal.values)):
            if value == True:
                signal[length - 1 - ii] = False
            else:
                break
    return signal

def get_int_signal(signal):
    int_signal = signal.astype(int).diff(1)
    return int_signal

def maxLimitClosed(signal, limit_unit):
    """
    :param signal(backtesting): Series [Boolean]
    :return: modified_signal: Series
    """
    assert signal[0] != True, "Signal not for backtesting"
    assert signal[len(signal) - 1] != True, "Signal not for backtesting"
    assert signal[len(signal) - 2] != True, "Signal not for backtesting"

    int_signal = get_int_signal(signal)
    signal_starts = [i - 1 for i in indexModel.get_open_index(int_signal)]
    signal_ends = [i - 1 for i in indexModel.get_close_index(int_signal)]
    starts, ends = indexModel.simple_limit_end_index(signal_starts, signal_ends, limit_unit)

    # assign new signal
    signal[:] = False
    for s, e in zip(starts, ends):
        signal[s:e] = True
    return signal

def get_MACD_signal(df, long_mode=True, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    :param period: int
    :param th: int: 0-100(+/-ve)
    :param upper: Boolean
    :return:
    """
    macd, macdsignal, macdhist = techModel.get_macd(df, fastperiod, slowperiod, signalperiod)
    if long_mode:
        signal = macd > 0
    else:
        signal = macd < 0
    return signal

def get_RSI_signal(df, period, th):
    """
    :param period: int
    :param th: int: 0-100(+/-ve)
    :param upper: Boolean
    :return:
    """
    rsi = techModel.get_rsi(df, period)
    if th > 0:
        signal = rsi >= abs(th)
    else:
        signal = rsi <= abs(th)
    return signal

def get_movingAverage_signal(df, fast_index, slow_index, limit_unit, long_mode=True, backtest=True):
    """
    :param slow_index: int
    :param fast_index: int
    :return: Series(Boolean)
    """
    fast = techModel.get_moving_average(df, fast_index)
    slow = techModel.get_moving_average(df, slow_index)
    if long_mode:
        signal = fast > slow
    else:
        signal = fast < slow
    if backtest: # discard if had ahead signal or tailed signal
        signal = discard_head_signal(signal)
        signal = discard_tail_signal(signal)
    if limit_unit > 0:
        signal = maxLimitClosed(signal, limit_unit)
    return signal

# from production.codes.lib.Trader import Data
# from production.codes.config import *
# mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
# df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)
# get_movingAverage_signal(df, 7,13,2)