from mt5Server.codes.Backtest import timeModel
import MetaTrader5 as mt5
import collections
import pandas as pd


class BaseMt5:
    def __init__(self):
        self.timeframe_dict = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
                               "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
                               "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
                               "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
                               "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
                               "MN1": mt5.TIMEFRAME_MN1}
        self.connect_server()
        print("MetaTrader 5 is connected. ")
        self.all_symbol_info = self.get_all_symbols_info()

    def get_symbol_total(self):
        """
        :return: int: number of symbols
        """
        num_symbols = mt5.symbols_total()
        if num_symbols > 0:
            print("Total symbols: ", num_symbols)
        else:
            print("Symbols not found.")
        return num_symbols

    def get_symbols(self, group=None):
        """
        :param group: https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolsget_py, refer to this website for usage of group
        :return: tuple(symbolInfo), there are several property
        """
        if group:
            symbols = mt5.symbols_get(group)
        else:
            symbols = mt5.symbols_get()
        return symbols

    def get_spread_from_ticks(self, ticks_frame, symbol):
        """
        :param ticks_frame: pd.DataFrame, all tick info
        :return: pd.Series
        """
        spread = pd.Series((ticks_frame['ask'] - ticks_frame['bid']) * (10 ** mt5.symbol_info(symbol).digits), index=ticks_frame.index, name='ask_bid_spread_pt')
        spread = spread.groupby(spread.index).mean()  # groupby() note 56b
        return spread

    def get_ticks_range(self, symbol, start, end, timezone):
        """
        :param symbol: str, symbol
        :param start: tuple, (2019,1,1)
        :param end: tuple, (2020,1,1)
        :param count:
        :return:
        """
        utc_from = timeModel.get_utc_time_from_broker(start, timezone)
        utc_to = timeModel.get_utc_time_from_broker(end, timezone)
        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        ticks_frame = pd.DataFrame(ticks)  # set to dataframe, several name of cols like, bid, ask, volume...
        ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')  # transfer numeric time into second
        ticks_frame = ticks_frame.set_index('time')  # set the index
        return ticks_frame

    def get_last_tick(self, symbol):
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

    def get_all_symbols_info(self):
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
                symbols_info[symbol_name].pt_value = 100  # 100 dollar for quote per each point    (See note Stock Market - Knowledge - note 3)
            else:
                symbols_info[symbol_name].pt_value = 1  # 1 dollar for quote per each point  (See note Stock Market - Knowledge - note 3)
        return symbols_info

    def connect_server(self):
        # connect to MetaTrader 5
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        else:
            print("MetaTrader Connected")

    def disconnect_server(self):
        # disconnect to MetaTrader 5
        mt5.shutdown()
        print("MetaTrader Shutdown.")

    def __enter__(self):
        self.connect_server()
        print("MetaTrader 5 is connected. ")

    def __exit__(self, *args):
        self.disconnect_server()
        print("MetaTrader 5 is disconnected. ")
