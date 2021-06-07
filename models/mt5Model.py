from production.codes import config
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import collections
from datetime import datetime, timedelta
import os
import pytz

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

class Helper:
    def __init__(self):
        self.text = ''
        self.text_line = 0

    def __enter__(self):
        connect_server()
        return self

    def __exit__(self, *args):
        disconnect_server()

    def append_dict_into_text(self, stat):
        """
        :param stat: dictionary {}
        :return: None
        """
        if self.text_line == 0:  # header only for first line
            for key in stat.keys():
                self.text += key + ','
            index = self.text.rindex(',')  # find the last index
            self.text = self.text[:index] + '\n'  # and replace
            self.text_line += 1
        for value in stat.values():
            self.text += str(value) + ','
        index = self.text.rindex(',')  # find the last index
        self.text = self.text[:index] + '\n'  # and replace
        self.text_line += 1

    def write_csv(self):
        print("\nFrame: {}\nLong Mode: {}\nFrom: {}\nTo: {}\n".format(str(config.TIMEFRAME_TEXT), str(config.LONG_MODE), config.START_STRING, config.END_STRING))
        print("Writing csv ... ", end='')
        with open(config.CSV_FILE_PATH + config.CSV_FILE_NAME, 'w') as f:
            f.write(self.text)
        print("OK")

class Trader:
    def __init__(self, dt_string, history_path, deviations, type_filling='ioc'):
        """
        :param type_filling: 'fok', 'ioc', 'return'
        :param deviation: int
        """
        self.history_path = history_path
        self.deviations = deviations
        self.type_filling = type_filling
        self.dt_string = dt_string
        self.history, self.record, self.status, self.strategy_symbols, self.order_ids = {}, {}, {}, {}, {} # see note 60b

    def __enter__(self):
        connect_server()
        self.all_symbol_info = get_all_symbols_info()
        return self

    def __exit__(self, *args):
        self.write_history_csv()
        disconnect_server()

    def write_history_csv(self):
        for strategy_id, history_df in self.history.items():
            if history_df != None:
                full_path = os.path.join(self.history_path, "{}_{}.csv".format(self.dt_string, strategy_id))
                history_df.to_csv(full_path)
            else:
                print("No histories being printed.")
        print("The histories are wrote to {}".format(self.history_path))

    def order_id_format(self, symbols):
        """
        update the order_id and it has container which is dictionary, note 59b
        :param strategy_id: str
        :return: dictionary
        """
        sub_order_ids = {}
        for symbol in symbols:
            sub_order_ids[symbol] = 'NO DEAL'
        return sub_order_ids

    def update_order_id(self, strategy_id, requests, results):
        """
        initialize the container dictionary, note 59b
        :param strategy_id: str
        :param requests: [request], mt5 request: https://www.mql5.com/en/docs/constants/structures/mqltraderequest
        :param results: [result], mt5 result: https://www.mql5.com/en/docs/constants/structures/mqltraderesult
        :return:
        """
        for request, result in zip(requests, results):
            self.order_ids[strategy_id][request['symbol']] = result.order

    def curr_record_format(self, symbols):
        """
        :param symbols: [str]
        :return: pd.DataFrame
        """
        column_index_arr = [
            np.array(['open'] * (len(symbols) + 1)  + ['close'] * (len(symbols) + 3)),
            np.array(['time', *symbols, 'time', *symbols, 'ret', 'earning'])    # Asterisk * unpacking the list, Programming/Python note 14
        ]
        return pd.DataFrame(columns=column_index_arr)

    def update_curr_record(self, strategy_id, results, expected_positions, ret, earning, open_pos):
        """
        :param strategy_id: str
        :param expected_positions: [float]
        :param ret: float
        :param earning: float
        :param open_pos: Boolean
        :return: None
        """
        # current time
        now = datetime.now()
        dt_string = now.strftime("%y%m%d%H%M%S")

        # judge if it is open position or close position
        if open_pos:
            type_position = 'open'
        else:
            # if it is close position, it need to fill the return and earning
            type_position = 'close'
            self.record[strategy_id].loc[0, (type_position, 'ret')] = ret
            self.record[strategy_id].loc[0, (type_position, 'earning')] = earning

        # data need to fill in for both open and close position
        self.record[strategy_id].loc[0, (type_position, 'time')] = dt_string
        for symbol, order_id, result, expect in zip(self.strategy_symbols[strategy_id], self.order_ids[strategy_id].values(), results, expected_positions):
            result_txt = ''
            real = result.price
            diff = expect - real
            if diff >= 0: result_txt = "{:.5f}+{} ({})".format(expect, diff, order_id)
            elif diff < 0: result_txt = "{:.5f}-{} ({})".format(expect, diff, order_id)
            self.record[strategy_id].loc[0, (type_position, symbol)] = result_txt

    def update_record_history_status(self, strategy_id, results, expected_positions, ret=0, earning=0):
        """
        :param strategy_id: str
        :param results: mt5 results object
        :param expected_positions: [float]
        :param ret: float
        :param earning: float
        :param open_pos: Boolean
        :return:
        """
        if self.status[strategy_id] == 0:
            self.update_curr_record(strategy_id, results, expected_positions, ret=ret, earning=earning, open_pos=True)
            self.status[strategy_id] = 1
        elif self.status[strategy_id] == 1:
            self.update_curr_record(strategy_id, results, expected_positions, ret=ret, earning=earning, open_pos=False)
            if self.history[strategy_id].empty:
                self.history[strategy_id] = self.record[strategy_id].copy()
            else:
                self.history[strategy_id] = pd.concat([self.history[strategy_id], self.record[strategy_id]], axis=0)
            self.record[strategy_id] = self.curr_record_format(self.strategy_symbols[strategy_id]) # format the current record after append the record
            self.status[strategy_id] = 0
        # init order id dictionary at every deal
        self.order_ids[strategy_id] = self.order_id_format(self.strategy_symbols[strategy_id])

    def register_strategy(self, strategy_id, symbols):
        """
        :param strategy_id: str
        :param symbols: [str]
        :param lots: [float], that is lots of open position. If close the position, product with negative 1
        :return: None
        """
        self.status[strategy_id] = 0 # 1 = Holding, 0 = None
        self.strategy_symbols[strategy_id] = symbols
        self.history[strategy_id] = pd.DataFrame()
        self.record[strategy_id] = self.curr_record_format(self.strategy_symbols[strategy_id])
        self.order_ids[strategy_id] = self.order_id_format(self.strategy_symbols[strategy_id])

    def check_allowed_with_deviation(self, requests):
        """
        if condition cannot meet, return False
        :param strategy_id: str
        :param lots: [float], that is lots going to buy(+ve) / sell(-ve)
        :param prices_at: [float]
        :return: Boolean
        """
        for request in requests:
            symbol, price_at, deviation, lot = request['symbol'], request['price'], request['deviation'], request['volume']
            if lot > 0:
                cost_price = mt5.symbol_info_tick(symbol).ask
                diff_pt = (cost_price - price_at) * (10 ** self.all_symbol_info[symbol].digits)
                if diff_pt > deviation:
                    print("{} have large deviation. {:.5f}(ask) - {:.5f}(price_at) = {:.3f}".format(symbol, cost_price, price_at, diff_pt))
                    return False
            else:
                cost_price = mt5.symbol_info_tick(symbol).bid
                diff_pt = (price_at - cost_price) * (10 ** self.all_symbol_info[symbol].digits)
                if diff_pt > deviation:
                    print("{} have large deviation. {:.5f}(price_at) - {:.5f}(bid) = {:.3f}".format(symbol, price_at, cost_price, diff_pt))
                    return False
        return True

    def strategy_execute(self, strategy_id, lots, prices_at):
        """
        :param strategy_id: str
        :param lots: [float], that is lots going to buy(+ve) / sell(-ve)
        :param prices_at: [float]
        :param pos_open_close: Boolean
        :return: Boolean
        """
        requests = self.requests_format(strategy_id, lots, prices_at)
        deviation_allowed = self.check_allowed_with_deviation(requests) # note 59a
        if not deviation_allowed:
            return False
        results = self.requests_execute(requests)
        if not results:
            return False
        self.update_order_id(strategy_id, requests, results)
        return results

    def requests_format(self, strategy_id, lots, prices_at):
        """
        :param strategy_id: str, belong to specific strategy
        :param lots: [float]
        :param prices_at: [float]
        :return: requests, [dict], a list of request
        """
        # the target with respect to the strategy id
        symbols = self.strategy_symbols[strategy_id]

        # type of filling
        tf = None
        if self.type_filling == 'fok':
            tf = mt5.ORDER_FILLING_FOK
        elif self.type_filling == 'ioc':
            tf = mt5.ORDER_FILLING_IOC
        elif self.type_filling == 'return':
            tf = mt5.ORDER_FILLING_RETURN
        # bui;ding each request
        requests = []
        for symbol, lot, price, deviation in zip(symbols, lots, prices_at, self.deviations):
            if lot > 0:
                action_type = mt5.ORDER_TYPE_BUY
            elif lot < 0:
                action_type = mt5.ORDER_TYPE_SELL
                lot = -lot
            else:
                continue # if lot equal to 0, do not append the request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': float(lot),
                'type': action_type,
                'price': price,
                'deviation': deviation,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": tf,
            }
            requests.append(request)
        return requests

    def requests_execute(self, requests):
        """
        :param requests: [request]
        :return: Boolean
        """
        # execute the request first and store the results
        results = []
        for request in requests:
            result = mt5.order_send(request)  # sending the request
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("order_send failed, symbol={}, retcode={}".format(request['symbol'], result.retcode))
                return False
            results.append(result)
        # print the results
        for request, result in zip(requests, results):
            print("order_send(): by {} {:.2f} lots at {} (ptDiff={:.1f} ({}[expected] - {}[real]))".format(
                request['symbol'], result.volume, result.price,
                (request['price'] - result.price) * 10 ** mt5.symbol_info(request['symbol']).digits,
                request['price'], result.price)
            )
        return results

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

def get_utc_time(time, timezone):
    """
    :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :return: datetime format
    """
    tz = pytz.timezone(timezone)
    utc_time = datetime(time[0], time[1], time[2], hour=time[3], minute=time[4], tzinfo=tz) + timedelta(hours=config.BROKER_TIME_BETWEEN_UTC, minutes=0)
    return utc_time

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

def get_spread_from_ticks(ticks_frame, symbol):
    """
    :param ticks_frame: pd.DataFrame, all tick info
    :return: pd.Series
    """
    spread = pd.Series((ticks_frame['ask'] - ticks_frame['bid']) * (10 ** mt5.symbol_info(symbol).digits), index=ticks_frame.index, name='ask_bid_spread_pt')
    spread = spread.groupby(spread.index).mean()    # groupby() note 56b
    return spread

def get_ticks_range(symbol, start, end, timezone):
    """
    :param symbol: str, symbol
    :param start: tuple, (2019,1,1)
    :param end: tuple, (2020,1,1)
    :param count:
    :return:
    """
    utc_from = get_utc_time(start, timezone)
    utc_to = get_utc_time(end, timezone)
    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    ticks_frame = pd.DataFrame(ticks) # set to dataframe, several name of cols like, bid, ask, volume...
    ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s') # transfer numeric time into second
    ticks_frame = ticks_frame.set_index('time') # set the index
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
            symbols_info[symbol_name].pt_value = 100   # 100 dollar for quote per each point    (See note Stock Market - Knowledge - note 3)
        else:
            symbols_info[symbol_name].pt_value = 1     # 1 dollar for quote per each point  (See note Stock Market - Knowledge - note 3)
    return symbols_info