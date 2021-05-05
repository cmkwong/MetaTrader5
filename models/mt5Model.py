import pandas as pd
import MetaTrader5 as mt5
import pytz
from datetime import datetime
from production.codes.controllers import mt5Controller

def get_timeframe(timeframe_txt):
    timeframe_dicts = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
                      "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10,
                      "M12": mt5.TIMEFRAME_M12,
                      "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
                      "H1": mt5.TIMEFRAME_H1,
                      "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
                      "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
                      "MN1": mt5.TIMEFRAME_MN1}
    return timeframe_dicts[timeframe_txt]

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

def get_prices_matrix(start, end, symbols, timeframe, timezone):
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
            price = get_historical_data(start, end, symbol, timeframe, timezone)
            price = price.set_index('time')['close']
            if i == 0:
                price_matrix = price
            else:
                price_matrix = pd.concat([price_matrix, price], axis=1, join='inner')
    return price_matrix.values.reshape(len(price_matrix), -1)