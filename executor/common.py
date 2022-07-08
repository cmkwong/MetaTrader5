from backtest import timeModel

import MetaTrader5 as mt5
import collections
import pandas as pd


def connect_server():
    # connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    else:
        print("MetaTrader Connected")


def disconnect_server():
    # disconnect to MetaTrader 5
    mt5.shutdown()
    print("MetaTrader Shutdown.")


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


def get_spread_from_ticks(ticks_frame, symbol):
    """
    :param ticks_frame: pd.DataFrame, all tick info
    :return: pd.Series
    """
    spread = pd.Series((ticks_frame['ask'] - ticks_frame['bid']) * (10 ** mt5.symbol_info(symbol).digits), index=ticks_frame.index, name='ask_bid_spread_pt')
    spread = spread.groupby(spread.index).mean()  # groupby() note 56b
    return spread


def get_ticks_range(symbol, start, end, timezone):
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
            symbols_info[symbol_name].pt_value = 100  # 100 dollar for quote per each point    (See note Stock Market - Knowledge - note 3)
        else:
            symbols_info[symbol_name].pt_value = 1  # 1 dollar for quote per each point  (See note Stock Market - Knowledge - note 3)
    return symbols_info


class BaseMt5:
    def __init__(self, type_filling='ioc'):
        self.type_filling = type_filling

    def __enter__(self):
        connect_server()
        self.all_symbol_info = get_all_symbols_info()
        print("MetaTrader 5 is connected. ")

    def __exit__(self, *args):
        disconnect_server()
        print("MetaTrader 5 is disconnected. ")

    def request_format(self, symbol, deviation, lot):
        """
        :param strategy_id: str, belong to specific strategy
        :param lots: [float]
        :param close_pos: Boolean, if it is for closing position, it will need to store the position id for reference
        :return: requests, [dict], a list of request
        """

        # type of filling
        tf = None
        if self.type_filling == 'fok':
            tf = mt5.ORDER_FILLING_FOK
        elif self.type_filling == 'ioc':
            tf = mt5.ORDER_FILLING_IOC
        elif self.type_filling == 'return':
            tf = mt5.ORDER_FILLING_RETURN

        # building request format
        if lot > 0:
            action_type = mt5.ORDER_TYPE_BUY  # int = 0
            price = mt5.symbol_info_tick(symbol).ask
        elif lot < 0:
            action_type = mt5.ORDER_TYPE_SELL  # int = 1
            price = mt5.symbol_info_tick(symbol).bid
            lot = -lot
        else:
            raise Exception("The lot cannot be 0")  # if lot equal to 0, raise an Error
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(lot),
            'type': action_type,
            'price': price,
            'deviation': deviation,  # indeed, the deviation is useless when it is marketing order, note 73d
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": tf,
        }
        return request

    def request_execute(self, request):
        """
        :param request: request
        :return: Boolean
        """
        result = mt5.order_send(request)  # sending the request
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("order_send failed, symbol={}, retcode={}".format(request['symbol'], result.retcode))
            return False
        print(
            f"Action: {request['type']}; by {request['symbol']} {result.volume:.2f} lots at {result.price:.5f} ( ptDiff={((request['price'] - result.price) * 10 ** mt5.symbol_info(request['symbol']).digits):.1f} ({request['price']:.5f}(request.price) - {result.price:.5f}(result.price) ))")
        return True