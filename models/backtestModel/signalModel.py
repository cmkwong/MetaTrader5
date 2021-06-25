from production.codes.models.backtestModel import techModel, indexModel
import pandas as pd
from datetime import timedelta

def get_latest_signal(signal, latest_index):
    """
    :param signal:
    :param latest_index:
    :return:
    """
    latest_signal = pd.Series(False, index=latest_index)
    int_signal = get_int_signal(signal)
    open_index = indexModel.get_signal_start_index(int_signal.reset_index(drop=True))[-1]
    latest_signal.iloc[open_index:] = True
    return latest_signal

def discard_head_tail_signal(signal):
    """
    :param signal: pd.Series
    :return: signal: pd.Series
    """
    # head
    if signal[0] == True:
        for index, value in signal.items():
            if value == True:
                signal[index] = False
            else:
                break

    # tail
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
    """
    :param signal: pd.Series()
    :return: pd.Series(), int_signal
    """
    int_signal = signal.astype(int).diff(1)
    return int_signal

def maxLimitClosed(signal, limit_unit):
    """
    :param signal(backtesting): pd.Series [Boolean]
    :param limit_unit: int
    :return: modified_signal: pd.Series
    """
    assert signal[0] != True, "Signal not for backtesting"
    assert signal[len(signal) - 1] != True, "Signal not for backtesting"
    assert signal[len(signal) - 2] != True, "Signal not for backtesting"

    int_signal = get_int_signal(signal)
    signal_starts = [i - 1 for i in indexModel.get_open_index(int_signal.reset_index(drop=True))]
    signal_ends = [i - 1 for i in indexModel.get_close_index(int_signal.reset_index(drop=True))]
    starts, ends = indexModel.simple_limit_end_index(signal_starts, signal_ends, limit_unit)

    # assign new signal
    signal.iloc[:] = False
    for s, e in zip(starts, ends):
        signal.iloc[s:e] = True
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

def get_movingAverage_signal(long_ma_data, short_ma_data, limit_unit):
    """
    :param ma_data:
    :param limit_unit: int
    :param long_mode:
    :return:
    """
    long_signal = pd.Series(long_ma_data['fast'] > long_ma_data['slow'], index=long_ma_data.index)
    short_signal = pd.Series(short_ma_data['fast'] < short_ma_data['slow'], index=short_ma_data.index)
    long_signal = discard_head_tail_signal(long_signal) # see 40c
    short_signal = discard_head_tail_signal(short_signal)
    if limit_unit > 0:
        long_signal = maxLimitClosed(long_signal, limit_unit)
        short_signal = maxLimitClosed(short_signal, limit_unit)
    return long_signal, short_signal

def get_coin_NN_signal(coin_NN_data, upper_th, lower_th, discard=True):
    """
    this function can available for coinNN and coin model
    :param coin_NN_data: pd.Dataframe(), columns='real','predict','spread','z_score'
    :param upper_th: float
    :param lower_th: float
    :return: pd.Series() for long and short
    """
    long_signal = pd.Series(coin_NN_data['z_score'].values < lower_th, index=coin_NN_data.index, name='long_signal')
    short_signal = pd.Series(coin_NN_data['z_score'].values > upper_th, index=coin_NN_data.index, name='short_signal')
    if discard:
        long_signal = discard_head_tail_signal(long_signal) # see 40c
        short_signal = discard_head_tail_signal(short_signal)
    return long_signal, short_signal

def get_resoluted_signal(signal, index):
    """
    :param signal: pd.Series
    :param index: pd.DateTimeIndex / str in time format
    :param freq_step: the time step in hour
    :return:
    """
    # resume to datetime index
    signal.index = pd.to_datetime(signal.index)
    index = pd.to_datetime(index)

    # get int signal and its start_indexes and end_indexes
    int_signal = get_int_signal(signal)
    start_indexes = indexModel.get_signal_start_index(int_signal)
    end_indexes = indexModel.get_signal_end_index(int_signal)
    # start_indexes = pd.to_datetime(signal[signal==True].index)
    # end_indexes = pd.to_datetime(signal[signal==True].index).shift(freq_step, freq='H').shift(-1, freq='min') # note 82e

    # init the empty signal series
    resoluted_signal = pd.Series(False, index=index)
    for s, e in zip(start_indexes, end_indexes):
        e = e + timedelta(minutes=-1) # note 82e, use the timedelta to reduce 1 minute instead of shift()
        resoluted_signal.loc[s:e] = True
    return resoluted_signal