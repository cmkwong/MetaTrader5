from production.codes.models.backtestModel import returnModel, pointsModel
from production.codes.utils import tools
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

def _price_type_from_code(ohlc):
    """
    :param ohlc: str of code, eg: '1001'
    :return: list, eg: ['open', 'close']
    """
    type_names = ['open', 'high', 'low', 'close']
    required_types = []
    for i, c in enumerate(ohlc):
        if c == '1':
            required_types.append(type_names[i])
    return required_types

def _get_prices_df(symbols, timeframe, timezone, start, end, ohlc):
    required_types = _price_type_from_code(ohlc)
    prices_df = None
    for i, symbol in enumerate(symbols):
        price = get_historical_data(symbol, timeframe, timezone, start, end)
        price = price.set_index('time').loc[:,required_types]
        if i == 0:
            prices_df = price
        else:
            prices_df = pd.concat([prices_df, price], axis=1, join='inner')
    return prices_df

def get_Prices(symbols, all_symbols_info, timeframe, timezone, start, end=None, ohlc='1111', deposit_currency='USD'):
    """
    :param symbols: [str]
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: (2010,1,1,0,0)
    :param end:  (2020,1,1,0,0)
    :return: collection.nametuple - ohlc. They are pd.Dataframe
    """
    Prices_collection = collections.namedtuple("Prices_collection", ['o','h','l','c','ptDv','quote_exchg','base_exchg'])
    prices_df = _get_prices_df(symbols, timeframe, timezone, start, end, ohlc)

    # get point diff values
    diff_name = "ptDv"
    points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, prices_df['open'], all_symbols_info, temp_col_name=diff_name)

    # get the quote to deposit exchange rate
    q2d_name = "q2d"
    q2d_exchange_symbols = get_exchange_symbols(symbols, all_symbols_info, deposit_currency, type='q2d')
    q2d_exchange_rate_df = _get_prices_df(q2d_exchange_symbols, timeframe, timezone, start, end, ohlc='1000') # just need the open price
    q2d_exchange_rate_df, q2d_modified_names = modify_exchange_rate(symbols, q2d_exchange_symbols, q2d_exchange_rate_df,
                                                                   deposit_currency, type='q2d')
    q2d_exchange_rate_df.columns = [q2d_name] * len(q2d_exchange_symbols) # assign temp name

    # get the base to deposit exchange rate
    b2d_name = "b2d"
    b2d_exchange_symbols = get_exchange_symbols(symbols, all_symbols_info, deposit_currency, type='b2d')
    b2d_exchange_rate_df = _get_prices_df(b2d_exchange_symbols, timeframe, timezone, start, end, ohlc='1000')  # just need the open price
    b2d_exchange_rate_df, b2d_modified_names = modify_exchange_rate(symbols, b2d_exchange_symbols, b2d_exchange_rate_df,
                                                                   deposit_currency, type='b2d')
    b2d_exchange_rate_df.columns = [b2d_name] * len(b2d_exchange_symbols)  # assign temp name

    # joining two dataframe to get the consistent index
    prices_df = pd.concat([prices_df, points_dff_values_df, q2d_exchange_rate_df, b2d_exchange_rate_df], axis=1, join='inner')

    # assign the column into each collection tuple
    Prices = Prices_collection(o=prices_df.loc[:,'open'],
                               h=prices_df.loc[:,'high'],
                               l=prices_df.loc[:,'low'],
                               c=prices_df.loc[:,'close'],
                               ptDv=prices_df.loc[:,diff_name],
                               quote_exchg=prices_df.loc[:,q2d_name],
                               base_exchg=prices_df.loc[:,b2d_name])

    # re-assign the columns name
    for i, df in enumerate(Prices):
        if i < len(Prices) - 2:
            df.columns = symbols
        elif i == len(Prices) - 2:
            df.columns = q2d_modified_names
        elif i == len(Prices) - 1:
            df.columns = b2d_modified_names

    return Prices

def split_Prices(Prices, percentage):
    keys = list(Prices._asdict().keys())
    prices = collections.namedtuple("prices", keys)
    train_list, test_list = [], []
    for df in Prices:
        train, test = tools.split_df(df, percentage)
        train_list.append(train)
        test_list.append(test)
    Train_Prices = prices._make(train_list)
    Test_Prices = prices._make(test_list)
    return Train_Prices, Test_Prices

def modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df, deposit_currency, type='q2d'):
    """
    :param symbols:             ['AUDJPY', 'AUDUSD', 'CADJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    :param exchange_symbols:    ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD'] all is related to deposit currency
    :param exchange_rate_df: pd.DataFrame, the price from excahnge_symbols
    :param deposit_currency: "USD" / "GBP" / "EUR"
    :param type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: pd.DataFrame with cols name: ['JPYUSD', 'USD', 'JPYUSD', 'USD', 'USD', 'CADUSD']
    """
    symbol_new_names = []
    for i, exch_symbol in enumerate(exchange_symbols):
        base, quote = exch_symbol[:3], exch_symbol[3:]
        if type == 'q2d':
            if quote != deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:,i].values # inverse if it is eg: USDJPY
            elif quote == symbols[i][3:]:
                symbol_new_names.append("{}".format(deposit_currency))
                exchange_rate_df.iloc[:, i] = 1.0
        elif type == 'b2d':
            if base != deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))
            elif base == deposit_currency and exch_symbol != symbols[i]:
                symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values  # inverse if it is eg: USDCAD
            else:
                symbol_new_names.append("{}".format(deposit_currency))
                exchange_rate_df.iloc[:, i] = 1.0
    return exchange_rate_df, symbol_new_names

def get_exchange_symbols(symbols, all_symbols_info, deposit_currency='USD', type='q2d'):
    """
    Find all the currency pair related to and required currency and deposit symbol
    :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
    :param all_symbols_info: dict with nametuple
    :param deposit_currency: str: USD/GBP/EUR, main currency for deposit
    :param type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: [str], get required exchange symbol in list: ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    """
    symbol_names = list(all_symbols_info.keys())
    exchange_symbols = []
    target_symbol = None
    for symbol in symbols:
        if type == 'b2d': target_symbol = symbol[:3]
        elif type == 'q2d': target_symbol = symbol[3:]
        if target_symbol != deposit_currency:  # if the symbol not relative to required deposit currency
            test_symbol_1 = target_symbol + deposit_currency
            test_symbol_2 = deposit_currency + target_symbol
            if test_symbol_1 in symbol_names:
                exchange_symbols.append(test_symbol_1)
                continue
            elif test_symbol_2 in symbol_names:
                exchange_symbols.append(test_symbol_2)
                continue
            else: # if not found the relative pair with respect to deposit currency, raise the error
                raise Exception("{} has no relative currency with respect to deposit {}.".format(target_symbol, deposit_currency))
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