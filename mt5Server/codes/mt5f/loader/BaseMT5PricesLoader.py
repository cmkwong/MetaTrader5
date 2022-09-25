from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

from mt5Server.codes.mt5f import BaseMt5
from mt5Server.codes.backtest import timeModel
from mt5Server.codes.mt5f.loader import files


class BaseMT5PricesLoader:
    # the column name got from MT5
    type_names = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']

    def _get_symbol_info_tick(self, symbol):
        lasttick = mt5.symbol_info_tick(symbol)._asdict()
        return lasttick

    def _get_historical_data(self, symbol, timeframe, timezone, start, end=None):
        """
        :param symbol: str
        :param timeframe: str, '1H'
        :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
        :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
        :return: dataframe
        """
        timeframe = timeModel.get_txt2timeframe(timeframe)
        utc_from = timeModel.get_utc_time_from_broker(start, timezone)
        if end == None:  # if end is None, get the loader at current time
            now = datetime.today()
            now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
            utc_to = timeModel.get_utc_time_from_broker(now_tuple, timezone)
        else:
            utc_to = timeModel.get_utc_time_from_broker(end, timezone)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        rates_frame = pd.DataFrame(rates, dtype=float)  # create DataFrame out of the obtained loader
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')  # convert time in seconds into the datetime format
        rates_frame = rates_frame.set_index('time')
        return rates_frame

    def _get_current_bars(self, symbol, timeframe, count):
        """
        :param symbols: str
        :param timeframe: str, '1H'
        :param count: int
        :return: df
        """
        timeframe = timeModel.get_txt2timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # 0 means the current bar
        rates_frame = pd.DataFrame(rates, dtype=float)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame = rates_frame.set_index('time')
        return rates_frame

    def _price_type_from_code(self, ohlcvs):
        """
        :param ohlcvs: str of code, eg: '1001'
        :return: list, eg: ['open', 'close']
        """
        required_types = []
        for i, c in enumerate(ohlcvs):
            if c == '1':
                required_types.append(self.type_names[i])
        return required_types

    def _get_mt5_prices(self, symbols, timeframe, timezone, start=None, end=None, ohlcvs='111100', count: int = 10):
        """
        :param symbols: [str]
        :param timeframe: str, '1H'
        :param timezone: str "Hongkong"
        :param start: (2010,1,1,0,0), if both start and end is None, use function get_current_bars()
        :param end: (2020,1,1,0,0), if just end is None, get the historical loader from date to current
        :param ohlcvs: str, eg: '111100' => open, high, low, close, volume, spread
        :param count: int, for get_current_bar_function()
        :return: pd.DataFrame
        """
        join = 'outer'
        required_types = self._price_type_from_code(ohlcvs)
        prices_df = None
        for i, symbol in enumerate(symbols):
            if count > 0:  # get the latest units of loader
                price = self._get_current_bars(symbol, timeframe, count).loc[:, required_types]
                join = 'inner'  # if getting count, need to join=inner to check if loader getting completed
            elif count == 0:  # get loader from start to end
                price = self._get_historical_data(symbol, timeframe, timezone, start, end).loc[:, required_types]
            else:
                raise Exception('start-date must be set when end-date is being set.')
            if i == 0:
                prices_df = price.copy()
            else:
                prices_df = pd.concat([prices_df, price], axis=1, join=join)

        # replace NaN values with preceding values
        prices_df.fillna(method='ffill', inplace=True)
        prices_df.dropna(inplace=True, axis=0)

        # get prices in dict
        prices = self._prices_df2dict(prices_df, symbols, ohlcvs)

        return prices

    def _get_local_prices(self, data_path, symbols, data_time_difference_to_UTC, ohlcvs):
        """
        :param data_path: str
        :param symbols: [str]
        :param data_time_difference_to_UTC: int
        :param timeframe: str, eg: '1H', '1min'
        :param ohlcvs: str, eg: '1001'
        :return: pd.DataFrame
        """
        prices_df = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            print("Processing: {}".format(symbol))
            price_df = files.read_symbol_price(data_path, symbol, data_time_difference_to_UTC, ohlcvs=ohlcvs)
            if i == 0:
                prices_df = price_df.copy()
            else:
                # join='outer' method with all symbols in a bigger dataframe (axis = 1)
                prices_df = pd.concat([prices_df, price_df], axis=1, join='outer')  # because of 1 minute loader and for ensure the completion of loader, concat in join='outer' method

        # replace NaN values with preceding values
        prices_df.fillna(method='ffill', inplace=True)
        prices_df.dropna(inplace=True, axis=0)

        # get prices in dict
        prices = self._prices_df2dict(prices_df, symbols, ohlcvs)

        return prices

    def _prices_df2dict(self, prices_df, symbols, ohlcvs):

        # rename columns of the prices_df
        col_names = self._price_type_from_code(ohlcvs)
        prices_df.columns = col_names * len(symbols)

        prices = {}
        max_length = len(prices_df.columns)
        step = len(col_names)
        for i in range(0, max_length, step):
            symbol = symbols[int(i / step)]
            prices[symbol] = prices_df.iloc[:, i:i + step]
        return prices

    def _get_specific_from_prices(self, prices, required_symbols, ohlcvs):
        """
        :param prices: {symbol: pd.DataFrame}
        :param required_symbols: [str]
        :param ohlcvs: str, '1000'
        :return: pd.DataFrame
        """
        types = self._price_type_from_code(ohlcvs)
        required_prices = pd.DataFrame()
        for i, symbol in enumerate(required_symbols):
            if i == 0:
                required_prices = prices[symbol].loc[:, types].copy()
            else:
                required_prices = pd.concat([required_prices, prices[symbol].loc[:, types]], axis=1)
        required_prices.columns = required_symbols
        return required_prices

    def _get_ohlc_rule(self, df):
        """
        note 85e
        Only for usage on change_timeframe()
        :param check_code: list
        :return: raise exception
        """
        check_code = [0, 0, 0, 0]
        ohlc_rule = {}
        for key in df.columns:
            if key == 'open':
                check_code[0] = 1
                ohlc_rule['open'] = 'first'
            elif key == 'high':
                check_code[1] = 1
                ohlc_rule['high'] = 'max'
            elif key == 'low':
                check_code[2] = 1
                ohlc_rule['low'] = 'min'
            elif key == 'close':
                check_code[3] = 1
                ohlc_rule['close'] = 'last'
        # first exception
        if check_code[1] == 1 or check_code[2] == 1:
            if check_code[0] == 0 or check_code[3] == 0:
                raise Exception("When high/low needed, there must be open/close loader included. \nThere is not open/close loader.")
        # Second exception
        if len(df.columns) > 4:
            raise Exception("The DataFrame columns is exceeding 4")
        return ohlc_rule
