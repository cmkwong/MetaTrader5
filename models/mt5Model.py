import pandas as pd
import MetaTrader5 as mt5
import pytz
from datetime import datetime

def get_symbol_total():
    """
    :return: int: number of symbols
    """
    num_symbols = mt5.symbols_total()
    if num_symbols > 0:
        print("Total symbols: ", num_symbols)
    else:
        print("Symbols not found.")
    return num_symbols

def get_symbols(group=None):
    """
    :param group: https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolsget_py, refer to this website for usage of group
    :return: tuple(symbolInfo), there are several property
    """
    if group:
        symbols = mt5.symbols_get(group)
    else:
        symbols = mt5.symbols_get()
    return symbols

def get_last_tick(symbol):
    """
    :param symbol: str
    :return: dict: symbol info
    """
    # display the last GBPUSD tick
    lasttick = mt5.symbol_info_tick(symbol)
    # display tick field values in the form of a list
    last_tick_dict = lasttick._asdict()
    for key, value in last_tick_dict.items():
        print("  {}={}".format(key, value))
    return last_tick_dict

def get_historical_data(start, end, symbol, timeframe, timezone):
    """
    :param start: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param end: tuple (year, month, day, hour, mins)
    :param symbol: str
    :param timeframe: mt5.timeframe
    :return:
    """
    tz = pytz.timezone(timezone)
    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(start[0], start[1], start[2], hour=start[3], minute=start[4], tzinfo=tz)
    utc_to = datetime(end[0], end[1], end[2], hour=end[3], minute=end[4], tzinfo=tz)
    # get bars from USDJPY M5 within the interval of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    return rates_frame