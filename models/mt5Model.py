import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
import collections

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_txt2timeframe(timeframe_txt):
    timeframe_dicts = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
                      "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10,
                      "M12": mt5.TIMEFRAME_M12,
                      "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
                      "H1": mt5.TIMEFRAME_H1,
                      "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
                      "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
                      "MN1": mt5.TIMEFRAME_MN1}
    return timeframe_dicts[timeframe_txt]

def get_timeframe2txt(mt5_timeframe_txt):
    timeframe_dicts = {mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M2: "M2", mt5.TIMEFRAME_M3: "M3", mt5.TIMEFRAME_M4: "M4",
                      mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M6: "M6", mt5.TIMEFRAME_M10: "M10",
                      mt5.TIMEFRAME_M12: "M12",
                      mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M20: "M20", mt5.TIMEFRAME_M30: "M30",
                      mt5.TIMEFRAME_H1: "H1",
                      mt5.TIMEFRAME_H2: "H2", mt5.TIMEFRAME_H3: "H3", mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H6: "H6",
                      mt5.TIMEFRAME_H8: "H8", mt5.TIMEFRAME_H12: "H12", mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "D1",
                      mt5.TIMEFRAME_MN1: "MN1"}
    return timeframe_dicts[mt5_timeframe_txt]

def get_time_string(tt):
    """
    :param tt: time_tuple: tuple (yyyy,m,d,h,m) 
    :return: string
    """
    time_string = str(tt[0]) + '-' + str(tt[1]).zfill(2) + '-' + str(tt[2]).zfill(2) + '-' + str(tt[3]).zfill(2) + '-' + str(tt[4]).zfill(2)
    return time_string

def get_current_time_string():
    now = datetime.today()
    end_str = get_time_string((now.year, now.month, now.day, now.hour, now.minute))
    return end_str

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

def get_historical_data(symbol, timeframe, timezone, start, end=None, utc_diff=3):
    """
    :param symbol: str
    :param timeframe: mt5.timeframe
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param end (local time): tuple (year, month, day, hour, mins), if None, then take data until present
    :param utc_diff: difference between IC market (UTC+3) and UTC = 3
    :return: dataframe
    """
    tz = pytz.timezone(timezone)
    utc_from = datetime(start[0], start[1], start[2], hour=start[3], minute=start[4], tzinfo=tz) + timedelta(hours=utc_diff, minutes=0)
    if end == None:
        now = datetime.today()
        utc_to = datetime(now.year, now.month, now.day, hour=now.hour, minute=now.minute, tzinfo=tz) + timedelta(hours=utc_diff,                                                                                            minutes=0)
    else:
        utc_to = datetime(end[0], end[1], end[2], hour=end[3], minute=end[4], tzinfo=tz) + timedelta(hours=utc_diff, minutes=0)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    rates_frame = pd.DataFrame(rates, dtype=float) # create DataFrame out of the obtained data
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s') # convert time in seconds into the datetime format
    return rates_frame

def get_prices_df(symbols, timeframe, timezone, start, end=None):
    """
    :param start: (2010,1,1,0,0)
    :param end:  (2020,1,1,0,0)
    :param symbols: [str]
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :return: arr
    """
    prices_df = None
    for i, symbol in enumerate(symbols):
        price = get_historical_data(symbol, timeframe, timezone, start, end)
        price = price.set_index('time')['close'].rename(symbol)
        if i == 0:
            prices_df = price
        else:
            prices_df = pd.concat([prices_df, price], axis=1, join='inner')
    return prices_df



def get_return(prices_matrix, coefficient_vector):
    """
    :param prices_matrix: np.array, size = (total_len, feature_size)
    :param coefficient_vector: np.array, size = (feature_size, 1)
    :return: np.array, size = (total_len, )
    """
    prices_matrix = prices_matrix.reshape(len(prices_matrix), -1)
    coefficient_vector = coefficient_vector.reshape(len(coefficient_vector),1)
    ret = np.dot(prices_matrix, coefficient_vector).reshape(-1,)
    return ret

def append_coin_signal(plt_df, upper_th, lower_th):
    plt_df['short_spread'] = plt_df['z_score'].values > upper_th
    plt_df['long_spread'] = plt_df['z_score'].values < lower_th
    return plt_df

def append_points_dff_df(plt_df, symbols, all_symbols_info):
    """
    :param plt_df: pd.Dataframe
    :param symbols: [str]
    :param all_symbols_info: tuple, mt5.symbols_get(). The info including the digits.
    :return: plt_df, new pd.Dataframe that appended points change
    """
    for symbol in symbols:
        digits = all_symbols_info[symbol].digits - 1
        col_name = 'pt_' + symbol
        plt_df[col_name] = (plt_df[symbol] - plt_df[symbol].shift(periods=1)) * 10 ** (digits)
    return plt_df

def append_exchange_rate_df(plt_df, exchange_symbols, timeframe, timezone, start, end=None, deposit_currency='USD'):
    """
    :param plt_df: pd.Dataframe
    :param exchange_symbols:
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: tuple, eg: (2019)
    :param end: tuple, None = data till now
    :param deposit_currency: str, default USD
    :return: plt_df, new pd.Dataframe that appended exchange rate
    """
    exchange_rate_df = get_prices_df(exchange_symbols, timeframe, timezone, start, end)
    symbol_new_names = []
    for i, symbol in enumerate(exchange_symbols):
        if symbol[3:] != deposit_currency:
            symbol_new_names.append("exchg_" + symbol[3:] + symbol[:3])
            exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:,i].values
        else:
            symbol_new_names.append("exchg_" + symbol)
    exchange_rate_df.columns = symbol_new_names
    plt_df = pd.concat([plt_df, exchange_rate_df], axis=1, join='inner')    # inner join the dataframe
    return plt_df

def get_exchange_symbols(symbols, all_symbols_info, deposit_currency='USD'):
    """
    :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
    :param all_symbols_info: dict with nametuple
    :param deposit_currency: str: USD/GBP/EUR, main currency for deposit
    :return: [str], get required exchange symbol in list
    """
    symbol_names = list(all_symbols_info.keys())
    exchange_symbols = []
    for symbol in symbols:
        if symbol[3:] != deposit_currency:  # if the symbol not relative to deposit currency
            test_symbol_1 = symbol[3:] + deposit_currency
            test_symbol_2 = deposit_currency + symbol[3:]
            if test_symbol_1 in symbol_names:
                exchange_symbols.append(test_symbol_1)
                continue
            elif test_symbol_2 in symbol_names:
                exchange_symbols.append(test_symbol_2)
                continue
            else: # if not found the relative pair with respect to deposit currency, raise the error
                raise Exception("{} has no relative currency pair with respect to {}.".format(symbol, filter))
        else: # if the symbol already relative to deposit currency
            exchange_symbols.append(symbol)
    return exchange_symbols

def get_all_symbols_info():
    """
    :return: dict[symbol] = collections.nametuple
    """
    symbols_info = {}
    symbols = mt5.symbols_get()
    for symbol in symbols:
        symbol_name = symbol.name
        symbols_info[symbol_name] = collections.namedtuple("info", ['digits', 'base', 'quote', 'swap_long', 'swap_short', 'pt_value'])
        symbols_info[symbol_name].digits = symbol.digits
        symbols_info[symbol_name].base = symbol.currency_base
        symbols_info[symbol_name].quote = symbol.currency_profit
        symbols_info[symbol_name].swap_long = symbol.swap_long
        symbols_info[symbol_name].swap_short = symbol.swap_short
        if symbol_name[3:] == 'JPY':
            symbols_info[symbol_name].pt_value = 1000   # 1000 dollar for quote per each point
        else:
            symbols_info[symbol_name].pt_value = 10     # 10 dollar for quote per each point
    return symbols_info

# # establish connection to the MetaTrader 5 terminal
# if not mt5.initialize():
#     print("initialize() failed, error code =",mt5.last_error())
#     quit()
#
# symbols = ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
#
# mt5.shutdown()
# print()