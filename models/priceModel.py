from production.codes.models import mt5Model, timeModel
from production.codes.models.backtestModel import pointsModel
from production.codes.utils import tools
import collections
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

def get_historical_data(symbol, timeframe, timezone, start, end=None):
    """
    :param symbol: str
    :param timeframe: mt5.timeframe
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param end (local time): tuple (year, month, day, hour, mins), if None, then take data until present
    :return: dataframe
    """
    utc_from = timeModel.get_utc_time_from_broker(start, timezone)
    if end == None:
        now = datetime.today()
        now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
        utc_to = timeModel.get_utc_time_from_broker(now_tuple, timezone)
    else:
        utc_to = timeModel.get_utc_time_from_broker(end, timezone)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    rates_frame = pd.DataFrame(rates, dtype=float) # create DataFrame out of the obtained data
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s') # convert time in seconds into the datetime format
    rates_frame = rates_frame.set_index('time')
    return rates_frame

def get_current_bars(symbol, timeframe, count):
    """
    :param symbols:
    :param timeframe:
    :param count:
    :return:
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count) # 0 means the current bar
    rates_frame = pd.DataFrame(rates, dtype=float)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame = rates_frame.set_index('time')
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

def _get_prices_df(symbols, timeframe, timezone, start=None, end=None, ohlc='1111', count=10):
    """
    :param symbols: [str]
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: (2010,1,1,0,0), if both start and end is None, use function get_current_bars()
    :param end: (2020,1,1,0,0), if just end is None, get the historical data from date to current
    :param ohlc: str, eg: '1111'
    :param count: int, for get_current_bar_function()
    :return: pd.DataFrame
    """
    required_types = _price_type_from_code(ohlc)
    prices_df = None
    for i, symbol in enumerate(symbols):
        if start == None and end == None:
            price = get_current_bars(symbol, timeframe, count).loc[:, required_types]
        elif start != None:
            price = get_historical_data(symbol, timeframe, timezone, start, end).loc[:, required_types]
        else:
            raise Exception('start-date must be set when end-date is being set.')
        if i == 0:
            prices_df = price
        else:
            prices_df = pd.concat([prices_df, price], axis=1, join='inner')
    return prices_df

def get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, ohlc, count, exchg_type, col_names, start=None, end=None):
    """
    :param col_names:
    :param symbols: [str]
    :param all_symbols_info: mt5.symbols_info object
    :param deposit_currency: str, USD / GBP / EUR
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: (2010,1,1,0,0)
    :param end: (2020,1,1,0,0)
    :param ohlc: 'str', eg: '1000'
    :param count: int
    :param exchg_type: q2d = quote exchange to deposit, b2d = base exchange to deposit
    :param col_names: list, the name assigned to column names
    :return: pd.DataFrame
    """
    exchange_symbols = get_exchange_symbols(symbols, all_symbols_info, deposit_currency, exchg_type=exchg_type)
    exchange_rate_df = _get_prices_df(exchange_symbols, timeframe, timezone, start, end, ohlc=ohlc, count=count)  # just need the open price
    exchange_rate_df, modified_names = modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df,
                                                                    deposit_currency, exchg_type=exchg_type)
    exchange_rate_df.columns = col_names  # assign temp name
    return exchange_rate_df, modified_names

def get_Prices(symbols, timeframe, timezone, start=None, end=None, ohlc='1111', count=10, deposit_currency='USD'):
    """
    :param count:
    :param symbols: [str]
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: (2010,1,1,0,0)
    :param end:  (2020,1,1,0,0)
    :return: collection.nametuple - ohlc. They are pd.Dataframe
    """
    all_symbols_info = mt5Model.get_all_symbols_info()

    Prices_collection = collections.namedtuple("Prices_collection", ['o','h','l','c','ptDv','quote_exchg','base_exchg'])
    prices_df = _get_prices_df(symbols, timeframe, timezone, start, end, ohlc, count)

    # get point diff values
    diff_name = "ptDv"
    points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, prices_df['open'], prices_df['open'].shift(periods=1), all_symbols_info, col_names=[diff_name] * len(symbols))

    # get the quote to deposit exchange rate
    q2d_name = "q2d"
    q2d_exchange_rate_df, q2d_modified_names = get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, '1000', count, exchg_type=q2d_name, col_names=[q2d_name]*len(symbols), start=start, end=end)

    # get the base to deposit exchange rate
    b2d_name = "b2d"
    b2d_exchange_rate_df, b2d_modified_names = get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, '1000', count, exchg_type=b2d_name, col_names=[b2d_name]*len(symbols), start=start, end=end)

    # inner joining two dataframe to get the consistent index
    prices_df = pd.concat([prices_df, points_dff_values_df, q2d_exchange_rate_df, b2d_exchange_rate_df], axis=1, join='inner')

    # assign the column into each collection tuple
    Prices = Prices_collection(o=pd.DataFrame(prices_df.loc[:,'open'], index=prices_df.index),
                               h=pd.DataFrame(prices_df.loc[:,'high'], index=prices_df.index),
                               l=pd.DataFrame(prices_df.loc[:,'low'], index=prices_df.index),
                               c=pd.DataFrame(prices_df.loc[:,'close'], index=prices_df.index),
                               ptDv=pd.DataFrame(prices_df.loc[:,diff_name], index=prices_df.index),
                               quote_exchg=pd.DataFrame(prices_df.loc[:,q2d_name], index=prices_df.index),
                               base_exchg=pd.DataFrame(prices_df.loc[:,b2d_name], index=prices_df.index))

    # re-assign the columns name
    for i, df in enumerate(Prices):
        if i < len(Prices) - 2:
            df.columns = symbols
        elif i == len(Prices) - 2:
            df.columns = q2d_modified_names
        elif i == len(Prices) - 1:
            df.columns = b2d_modified_names

    return Prices

def get_latest_Prices(all_symbols_info, symbols, timeframe, timezone, count=10, deposit_currency='USD'):

    Prices_collection = collections.namedtuple("Prices_collection", ['c', 'l_o', 'l_ptDv', 'l_quote_exchg'])

    # get latest open prices and close prices
    prices_df = _get_prices_df(symbols, timeframe, timezone, ohlc='1001', count=count)
    if len(prices_df) != count:  # note 63a
        print("prices_df length of Data is not equal to count")
        return False
    new_index = prices_df.loc[:, 'open'].index.union([timeModel.get_current_utc_time_from_broker(timezone)]) # plus 1 length of data
    latest_open_price_arr = np.concatenate((prices_df['open'].values, prices_df['close'].values[-1,:].reshape(1,-1)), axis=0)
    latest_open_prices_df = pd.DataFrame(latest_open_price_arr, columns=['open'] * len(symbols), index=new_index)
    latest_close_price_arr = np.concatenate((prices_df['close'].values, prices_df['close'].values[-1, :].reshape(1, -1)), axis=0)
    latest_close_prices_df = pd.DataFrame(latest_close_price_arr, columns=['close'] * len(symbols), index=new_index)

    # get point diff values with latest value
    diff_name = "ptDv"
    points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, latest_open_prices_df, latest_open_prices_df.shift(periods=1), all_symbols_info, col_names=[diff_name] * len(symbols))

    # get quote exchange with values
    q2d_name = "q2d"
    q2d_exchange_rate_df_o, q2d_modified_names = get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, '1000', count, exchg_type=q2d_name, col_names=[q2d_name]*len(symbols))
    q2d_exchange_rate_df_c, _ = get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, '0001', count, exchg_type=q2d_name, col_names=[q2d_name]*len(symbols))
    # TODO if len(q2d_exchange_rate_df_o) or len(q2d_exchange_rate_df_c) == 39, return false and run again
    if len(q2d_exchange_rate_df_o) != count or len(q2d_exchange_rate_df_c) != count: # note 63a
        print("q2d_exchange_rate_df_o or q2d_exchange_rate_df_c length of Data is not equal to count")
        return False
    q2d_exchange_rate_arr = np.concatenate((q2d_exchange_rate_df_o.values, q2d_exchange_rate_df_c.values[-1, :].reshape(1, -1)), axis=0)
    q2d_exchange_rate_df = pd.DataFrame(q2d_exchange_rate_arr, columns=[q2d_name] * len(symbols), index=new_index)

    joined_prices_df = pd.concat([latest_open_prices_df, latest_close_prices_df, points_dff_values_df, q2d_exchange_rate_df], axis=1, join='inner')

    Prices = Prices_collection(c=joined_prices_df.loc[:,'close'].iloc[:-1,:], # discard the last row
                               l_o=pd.DataFrame(joined_prices_df.loc[:, 'open'], index=joined_prices_df.index),
                               l_ptDv=pd.DataFrame(points_dff_values_df.loc[:, diff_name], index=joined_prices_df.index),
                               l_quote_exchg=pd.DataFrame(q2d_exchange_rate_df.loc[:, q2d_name], index=joined_prices_df.index))

    # re-assign the columns name
    for i, df in enumerate(Prices):
        if i < len(Prices) - 1:
            df.columns = symbols
        elif i == len(Prices) - 1:
            df.columns = q2d_modified_names

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

def modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df, deposit_currency, exchg_type):
    """
    :param symbols:             ['AUDJPY', 'AUDUSD', 'CADJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    :param exchange_symbols:    ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD'] all is related to deposit currency
    :param exchange_rate_df: pd.DataFrame, the price from excahnge_symbols
    :param deposit_currency: "USD" / "GBP" / "EUR"
    :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: pd.DataFrame with cols name: ['JPYUSD', 'USD', 'JPYUSD', 'USD', 'USD', 'CADUSD']
    """
    symbol_new_names = []
    for i, exch_symbol in enumerate(exchange_symbols):
        base, quote = exch_symbol[:3], exch_symbol[3:]
        if exchg_type == 'q2d': # see note 38a
            if quote == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif base == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
        elif exchg_type == 'b2d':
            if base == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                    exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif quote == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))

    return exchange_rate_df, symbol_new_names

def get_exchange_symbols(symbols, all_symbols_info, deposit_currency='USD', exchg_type='q2d'):
    """
    Find all the currency pair related to and required currency and deposit symbol
    :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
    :param all_symbols_info: dict with nametuple
    :param deposit_currency: str: USD/GBP/EUR, main currency for deposit
    :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: [str], get required exchange symbol in list: ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    """
    symbol_names = list(all_symbols_info.keys())
    exchange_symbols = []
    target_symbol = None
    for symbol in symbols:
        if exchg_type == 'b2d': target_symbol = symbol[:3]
        elif exchg_type == 'q2d': target_symbol = symbol[3:]
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