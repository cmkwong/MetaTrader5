from production.codes.backtest import timeModel, pointsModel, returnModel
import csv
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import collections
import os

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

class csv_Writer_Helper:
    def __init__(self, csv_save_path='', csv_file_names=None, append_checkpoint=None):
        # for output csv file
        self.csv_save_path = csv_save_path
        self.append_checkpoint = append_checkpoint
        self._register_csv_file_datas(csv_file_names) # store the text, if no need to store for specific data file, return empty dictionary
        self._register_csv_txt_append_count(csv_file_names) # store the appending count for specific data file, if no need to append, return empty dictionary
        self._appended_text = False # if this class have been appended text, then set as True

    def __enter__(self):
        connect_server()
        return self

    def __exit__(self, *args):
        if self._appended_text: # if there is a appended text before, evacuate the rest of data before exit
            self.evacuate_csv_file_datas()
        disconnect_server()

    def _register_csv_file_datas(self, csv_file_names):
        self._csv_file_datas = {}
        if csv_file_names is not None:
            for csv_file_name in csv_file_names:
                self._csv_file_datas[csv_file_name] = ''

    def _register_csv_txt_append_count(self, csv_file_names):
        self._csv_txt_append_count = {}
        if csv_file_names is not None:
            for csv_file_name in csv_file_names:
                self._csv_txt_append_count[csv_file_name] = 0

    def append_txt(self, csv_text, csv_file_name):
        """
        :param csv_text: text being stored and write later
        :param csv_file_name: file name that being output or append
        """
        # check if csv file name is registered
        if csv_file_name not in list(self._csv_file_datas.keys()):
            raise Exception("{} is not registered.".format(csv_file_name))

        # if first time in appending text, do NOT discard the header
        if self._csv_txt_append_count[csv_file_name] > 0:
            csv_text = csv_text.split('\n', 1)[1] # discard first row that is header
        # append the text
        self._csv_file_datas[csv_file_name] += csv_text

        # if reach the checkpoint, write and empty the data as csv
        if self._csv_txt_append_count[csv_file_name] % self.append_checkpoint == 0:
            self.evacuate_csv_file_datas()
        self._csv_txt_append_count[csv_file_name] += 1
        self._appended_text = True

    def evacuate_csv_file_datas(self):
        for csv_file_name, csv_text in self._csv_file_datas.items():
            if len(csv_text) > 0:
                print("Writing {} ... ".format(csv_file_name), end='')
                with open(os.path.join(self.csv_save_path, csv_file_name), 'a+') as f:
                    f.write(csv_text)
                    f.close()
                    print('OK')
                # empty the datas
            self._csv_file_datas[csv_file_name] = ''

    # def append_dict_into_text(self, stat):
    #     """
    #     :param stat: dictionary {}
    #     :return: None
    #     """
    #     if self.text_line == 0:  # header only for first line
    #         for key in stat.keys():
    #             self.text += key + ','
    #         index = self.text.rindex(',')  # find the last index
    #         self.text = self.text[:index] + '\n'  # and replace
    #         self.text_line += 1
    #     for value in stat.values():
    #         self.text += str(value) + ','
    #     index = self.text.rindex(',')  # find the last index
    #     self.text = self.text[:index] + '\n'  # and replace
    #     self.text_line += 1

    # def add_header(self, csv_text, header_list):
    #     """
    #     :param csv_text: text for csv
    #     :param header_list: [str]
    #     :return:
    #     """
    #     return ','.join(header_list) + '\n' + csv_text
    #
    #
    # def write_csv(self, csv_text, csv_file_name):
    #     print("\nFrame: {} \nFrom: {}\nTo: {}\n".format(str(self.timeframe), self.start_time, self.end_time))
    #     print("Writing csv ... ", end='')
    #     with open(os.path.join(self.csv_save_path, csv_file_name), 'a') as f:
    #         f.write(csv_text)
    #     print("OK")

class Trader:
    def __init__(self, dt_string, history_path, type_filling='ioc'):
        """
        :param type_filling: 'fok', 'ioc', 'return'
        :param deviation: int
        """
        self.history_path = history_path
        self.type_filling = type_filling
        self.dt_string = dt_string
        self.history, self.status, self.strategy_symbols, \
        self.position_ids, self.deviations, self.avg_spreads, \
        self.open_postions, self.open_postions_date, self.close_postions, self.close_postions_date, \
        self.rets, self.earnings, self.mt5_deal_details, \
        self.q2d_at, self.open_point_diff, self.close_point_diff, self.lot_times, \
        self.long_modes = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} # see note 60b

    def __enter__(self):
        connect_server()
        self.all_symbol_info = get_all_symbols_info()
        return self

    def __exit__(self, *args):
        disconnect_server()

    def append_history_csv(self, strategy_id):
        history_df = self.history[strategy_id]
        full_path = os.path.join(self.history_path, "{}_{}.csv".format(self.dt_string, strategy_id))
        header = False
        if not os.path.isfile(full_path): header = True
        history_df.to_csv(full_path, mode='a', header=header)
        print("The histories are wrote to {}".format(full_path))

    def position_id_format(self, symbols):
        """
        update the order_id and it has container which is dictionary, note 59b
        :param strategy_id: str
        :return: dictionary
        """
        oid = {}
        for symbol in symbols:
            oid[symbol] = -1
        return oid

    def history_format(self, strategy_id):
        """
        :return: pd.DataFrame
        """
        symbols = self.strategy_symbols[strategy_id]
        level_2_arr = np.array(['ret', 'earning'] * 2 + ['commission', 'swap', 'fee', 'earning', 'balanced', 'diff', 'open_date', 'closed_date']
                               + symbols * 6)  # Asterisk * unpacking the list, Programming/Python note 14
        level_1_arr = np.array(['expected'] * 2 + ['real'] * 2 + ['mt5'] * 8 +
                               ['open_real'] * len(symbols) + ['close_real'] * len(symbols) +
                               ['open_expected'] * len(symbols) + ['close_expected'] * len(symbols) +
                               ['open_spread']*len(symbols) + ['close_spread']*len(symbols))
        column_index_arr = [
            level_1_arr, level_2_arr
        ]
        return pd.DataFrame(columns=column_index_arr)

    def mt5_deal_detail_format(self):
        deal_details = {}
        deal_details['commission'] = 0.0
        deal_details['swap'] = 0.0
        deal_details['fee'] = 0.0
        deal_details['earning'] = 0.0
        deal_details['balanced'] = 0.0
        return deal_details

    def update_position_id(self, strategy_id, results):
        """
        initialize the container dictionary, note 59b
        :param strategy_id: str
        :param requests: [request], mt5 request: https://www.mql5.com/en/docs/constants/structures/mqltraderequest
        :param results: [result], mt5 result: https://www.mql5.com/en/docs/constants/structures/mqltraderesult
        :return:
        """
        symbols = self.strategy_symbols[strategy_id]
        for i, result in enumerate(results):
            self.position_ids[strategy_id][symbols[i]] = result.order

    def update_history(self, strategy_id):
        dt_string = timeModel.get_current_time_string()
        dt = pd.to_datetime(dt_string, format='%Y-%m-%d-%H-%M', errors='ignore')
        # expected
        self.history[strategy_id].loc[dt, ('expected', 'ret')] = self.rets[strategy_id]['expected']
        self.history[strategy_id].loc[dt, ('expected', 'earning')] = self.earnings[strategy_id]['expected']
        # real
        self.history[strategy_id].loc[dt, ('real', 'ret')] = self.rets[strategy_id]['real']
        self.history[strategy_id].loc[dt, ('real', 'earning')] = self.earnings[strategy_id]['real']
        # mt5 deal details
        self.history[strategy_id].loc[dt, ('mt5', 'commission')] = self.mt5_deal_details[strategy_id]['commission']
        self.history[strategy_id].loc[dt, ('mt5', 'swap')] = self.mt5_deal_details[strategy_id]['swap']
        self.history[strategy_id].loc[dt, ('mt5', 'fee')] = self.mt5_deal_details[strategy_id]['fee']
        self.history[strategy_id].loc[dt, ('mt5', 'earning')] = self.mt5_deal_details[strategy_id]['earning']
        self.history[strategy_id].loc[dt, ('mt5', 'balanced')] = self.mt5_deal_details[strategy_id]['balanced']
        self.history[strategy_id].loc[dt, ('mt5', 'diff')] = self.mt5_deal_details[strategy_id]['balanced'] - self.earnings[strategy_id]['expected']
        # open date and closed date
        self.history[strategy_id].loc[dt, ('mt5', 'open_date')] = self.open_postions_date[strategy_id]
        self.history[strategy_id].loc[dt, ('mt5', 'closed_date')] = self.close_postions_date[strategy_id]

        # update spreads, note 74a
        for i, symbol in enumerate(self.strategy_symbols[strategy_id]):
            # open positions
            self.history[strategy_id].loc[dt, ('open_real', symbol)] = self.open_postions[strategy_id]['real'][i]
            self.history[strategy_id].loc[dt, ('open_expected', symbol)] = self.open_postions[strategy_id]['expected'][i]
            self.history[strategy_id].loc[dt, ('open_spread', symbol)] = self.open_point_diff[strategy_id][i]
            # close positions
            self.history[strategy_id].loc[dt, ('close_real', symbol)] = self.close_postions[strategy_id]['real'][i]
            self.history[strategy_id].loc[dt, ('close_expected', symbol)] = self.close_postions[strategy_id]['expected'][i]
            self.history[strategy_id].loc[dt, ('close_spread', symbol)] = self.close_point_diff[strategy_id][i]

        return True
    
    def update_mt5_deal_details(self, strategy_id):
        position_ids = self.position_ids[strategy_id]
        for position_id in position_ids.values():
            deals = mt5.history_deals_get(position=position_id)
            for deal in deals:
                self.mt5_deal_details[strategy_id]['commission'] += deal.commission
                self.mt5_deal_details[strategy_id]['swap'] += deal.swap
                self.mt5_deal_details[strategy_id]['fee'] += deal.fee
                self.mt5_deal_details[strategy_id]['earning'] += deal.profit
                self.mt5_deal_details[strategy_id]['balanced'] += deal.commission + deal.swap + deal.fee + deal.profit

    def register_strategy(self, strategy_id, symbols, deviations, avg_spreads, lot_times, long_mode):
        """
        :param strategy_id: str
        :param symbols: [str]
        :param lots: [float], that is lots of open position. If close the position, product with negative 1
        :return: None
        """
        self.status[strategy_id] = 0 # 1 = Holding, 0 = None
        self.strategy_symbols[strategy_id] = symbols
        self.deviations[strategy_id] = deviations
        self.avg_spreads[strategy_id] = avg_spreads
        self.lot_times[strategy_id] = lot_times
        self.open_postions_date[strategy_id] = False # see note 69a
        self.long_modes[strategy_id] = long_mode # Boolean
        self.init_strategy(strategy_id)

    def init_strategy(self, strategy_id):
        self.history[strategy_id] = self.history_format(strategy_id)
        self.position_ids[strategy_id] = self.position_id_format(self.strategy_symbols[strategy_id])
        self.open_postions[strategy_id], self.close_postions[strategy_id] = {}, {}
        self.rets[strategy_id], self.earnings[strategy_id] = {}, {}
        self.mt5_deal_details[strategy_id] = self.mt5_deal_detail_format()
        self.q2d_at[strategy_id] = np.zeros((len(self.strategy_symbols[strategy_id]),))
        # their difference in points
        self.open_point_diff[strategy_id] = np.zeros((len(self.strategy_symbols[strategy_id]),))
        self.close_point_diff[strategy_id] = np.zeros((len(self.strategy_symbols[strategy_id]),))

    def check_allowed_with_avg_spread(self, requests, expected_prices, avg_spreads):
        """
        check if the market is in very high spread, like hundred of point spread
        if condition cannot meet, return False
        :param requests: [dictionary]
        :param deviations: list
        :return: Boolean
        """
        for request, price_at, deviation in zip(requests, expected_prices, avg_spreads):
            symbol, action_type = request['symbol'], request['type']
            if action_type == mt5.ORDER_TYPE_BUY:
                cost_price = mt5.symbol_info_tick(symbol).ask
                diff_pt = (cost_price - price_at) * (10 ** self.all_symbol_info[symbol].digits)
                if diff_pt > deviation:
                    print("Buy {} have large deviation. {:.5f}(ask) - {:.5f}(price_at) = {:.3f}".format(symbol, cost_price, price_at, diff_pt))
                    return False
            elif action_type == mt5.ORDER_TYPE_SELL:
                cost_price = mt5.symbol_info_tick(symbol).bid
                diff_pt = (price_at - cost_price) * (10 ** self.all_symbol_info[symbol].digits)
                if diff_pt > deviation:
                    print("Sell {} have large deviation. {:.5f}(price_at) - {:.5f}(bid) = {:.3f}".format(symbol, price_at, cost_price, diff_pt))
                    return False
        return True

    def strategy_controller(self, strategy_id, close_prices, quote_exchg, coefficient_vector, signal, slsp, lots):
        """
        :param strategy_id: str, each strategy has unique id for identity
        :param close_prices: pd.DataFrame, open price with latest prices
        :param quote_exchg: pd.DataFrame, quote exchange rate with latest rate
        :param coefficient_vector: np.array (raw vector: [y-intercepy, coefficients])
        :param signal: pd.Series
        :param slsp: tuple, (stop-loss, stop-profit)
        :param lots: [float], that is lots of open position. If close the position, product with negative 1
        :return: None
        :param masked_open_prices: open price with last price masked by current price
        """
        # init
        results, requests = False, False

        different_open_position = (signal.index[-1] != self.open_postions_date[strategy_id])  # different position to the previous one, note 69a
        if signal[-2] == True and signal[-3] == False and self.status[strategy_id] == 0 and different_open_position:
            # if open signal has available
            expected_prices = close_prices.iloc[-2, :].values
            q2d_at = quote_exchg.iloc[-2, :].values
            print("\n----------------------------------{}: Open position----------------------------------".format(strategy_id))
            results, requests = self.strategy_open(strategy_id, expected_prices, lots)  # open position
            if results:
                self.strategy_open_update(strategy_id, results, requests, expected_prices, q2d_at, signal.index[-1])

        elif self.status[strategy_id] == 1:
            if signal[-2] == False and signal[-3] == True:
                expected_prices = close_prices.iloc[-2, :].values # -2 is the open price from the day
                expected_ret, expected_earning = returnModel.get_value_of_ret_earning(symbols=self.strategy_symbols[strategy_id],
                                                                                      new_values=expected_prices,
                                                                                      old_values=self.open_postions[strategy_id]['expected'],
                                                                                      q2d_at=self.q2d_at[strategy_id],
                                                                                      all_symbols_info=self.all_symbol_info,
                                                                                      lot_times=self.lot_times[strategy_id],
                                                                                      coefficient_vector=coefficient_vector,
                                                                                      long_mode=self.long_modes[strategy_id])
                print("\n----------------------------------{}: Close position----------------------------------".format(strategy_id))
                results, requests = self.strategy_close(strategy_id, lots)  # close position
                if results:
                    self.strategy_close_update(strategy_id, results, requests, coefficient_vector, expected_prices, expected_ret, expected_earning, signal.index[-1])
            else:
                expected_prices = close_prices.iloc[-1, :].values # -1 is latest value from the day
                expected_ret, expected_earning = returnModel.get_value_of_ret_earning(symbols=self.strategy_symbols[strategy_id],
                                                                                      new_values=expected_prices,
                                                                                      old_values=self.open_postions[strategy_id]['expected'],
                                                                                      q2d_at=self.q2d_at[strategy_id],
                                                                                      all_symbols_info=self.all_symbol_info,
                                                                                      lot_times=self.lot_times[strategy_id],
                                                                                      coefficient_vector=coefficient_vector,
                                                                                      long_mode=self.long_modes[strategy_id])
                print("ret: {}, earning: {}".format(expected_ret, expected_earning))
                print(str(expected_prices))
                if expected_earning > slsp[1]:  # Stop Profit
                    print("\n----------------------------------{}: Close position (Stop profit)----------------------------------".format(strategy_id))
                    results, requests = self.strategy_close(strategy_id, lots)  # close position
                elif expected_earning < slsp[0]:  # Stop Loss
                    print("\n----------------------------------{}: Close position (Stop Loss)----------------------------------".format(strategy_id))
                    results, requests = self.strategy_close(strategy_id, lots)  # close position
                if results:
                    self.strategy_close_update(strategy_id, results, requests, coefficient_vector, expected_prices, expected_ret, expected_earning, signal.index[-1])

    def strategy_open_update(self, strategy_id, results, requests, expected_prices, q2d_at, open_position_date):
        """
        :param strategy_id: str
        :param results: mt5 results
        :param requests: request dict
        :param expected_prices: np.array, size = (len(symbols), )
        :param q2d_at: np.array
        :param open_position_date: the date that open position
        :return: Boolean
        """
        # update status
        self.status[strategy_id] = 1
        # update the open position: expected
        self.open_postions[strategy_id]['expected'] = expected_prices
        # update the open position: real
        self.open_postions[strategy_id]['real'] = [result.price for result in results]
        # update open pt diff
        self.open_point_diff[strategy_id] = pointsModel.get_point_diff_from_results(results, requests, expected_prices, self.all_symbol_info)
        # date
        self.open_postions_date[strategy_id] = open_position_date # update the open position time to avoid the buy again after stop loss or profit
        self.q2d_at[strategy_id] = q2d_at
        return True

    def strategy_close_update(self, strategy_id, results, requests, coefficient_vector, expected_prices, expected_ret, expected_earning, close_position_date):
        """
        :param strategy_id: str
        :param results: mt5 results
        :param coefficient_vector: np.array
        :param expected_prices: np.array, size = (len(symbols), )
        :param expected_ret: float
        :param expected_earning: float
        :param long_mode: Boolean
        :param close_position_date: the date that close position
        :return: Boolean
        """
        # update closed date
        self.close_postions_date[strategy_id] = close_position_date
        # update the close position: expected
        self.close_postions[strategy_id]['expected'] = expected_prices
        self.rets[strategy_id]['expected'] = expected_ret
        self.earnings[strategy_id]['expected'] = expected_earning

        # update the close position: real
        real_close_prices = np.array([result.price for result in results])
        real_ret, real_earning = returnModel.get_value_of_ret_earning(symbols=self.strategy_symbols[strategy_id],
                                                                      new_values=real_close_prices,
                                                                      old_values=self.open_postions[strategy_id]['real'],
                                                                      q2d_at=self.q2d_at[strategy_id],
                                                                      all_symbols_info=self.all_symbol_info,
                                                                      lot_times=self.lot_times[strategy_id],
                                                                      coefficient_vector=coefficient_vector,
                                                                      long_mode=self.long_modes[strategy_id])
        self.close_postions[strategy_id]['real'] = real_close_prices
        self.rets[strategy_id]['real'] = real_ret
        self.earnings[strategy_id]['real'] = real_earning

        # update close pt diff
        self.close_point_diff[strategy_id] = pointsModel.get_point_diff_from_results(results, requests, expected_prices, self.all_symbol_info)

        # update status
        self.status[strategy_id] = 0

        # update history
        self.update_mt5_deal_details(strategy_id)
        self.update_history(strategy_id)

        # write csv file
        self.append_history_csv(strategy_id)

        # clear the all record including history
        self.init_strategy(strategy_id)  # clear the record
        return True

    def strategy_open(self, strategy_id, expected_prices, lots):
        """
        :param strategy_id: str
        :param lots: [float], that is open position that lots going to buy(+ve) / sell(-ve)
        :return: dict: requests, results
        """
        requests = self.requests_format(strategy_id, lots, close_pos=False)
        spread_allowed = self.check_allowed_with_avg_spread(requests, expected_prices, self.avg_spreads[strategy_id]) # note 59a
        if not spread_allowed:
            return False, False
        results = self.requests_execute(requests)

        # update the order id
        self.update_position_id(strategy_id, results)

        # if results is not completed in all positions
        if len(results) < len(self.strategy_symbols[strategy_id]):
            self.strategy_close(strategy_id, lots)
            print("{}: The open position is failed. The previous opened position are closed.".format(strategy_id))
            return False, False
        return results, requests

    def strategy_close(self, strategy_id, lots):
        """
        :param strategy_id: str
        :param lots: [float], that is open position that lots going to buy(+ve) / sell(-ve)
        :return: dict: requests, results
        """
        lots = [-l for l in lots]
        requests = self.requests_format(strategy_id, lots, close_pos=True)
        results = self.requests_execute(requests)
        return results, requests

    def requests_format(self, strategy_id, lots, close_pos=False):
        """
        :param strategy_id: str, belong to specific strategy
        :param lots: [float]
        :param close_pos: Boolean, if it is for closing position, it will need to store the position id for reference
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
        # building each request
        requests = []
        for symbol, lot, deviation in zip(symbols, lots, self.deviations[strategy_id]):
            if lot > 0:
                action_type = mt5.ORDER_TYPE_BUY # int = 0
                price = mt5.symbol_info_tick(symbol).ask
            elif lot < 0:
                action_type = mt5.ORDER_TYPE_SELL # int = 1
                price = mt5.symbol_info_tick(symbol).bid
                lot = -lot
            else:
                raise Exception("The lot cannot be 0") # if lot equal to 0, raise an Error
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': float(lot),
                'type': action_type,
                'price': price,
                'deviation': deviation, # indeed, the deviation is useless when it is marketing order, note 73d
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": tf,
            }
            if close_pos:
                if self.position_ids[strategy_id][symbol] == -1:
                    continue    # if there is no order id, do not append the request, note 63b (default = 0)
                request['position'] = self.position_ids[strategy_id][symbol] # note 61b
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
                return results
            results.append(result)
        # print the results
        for request, result in zip(requests, results):
            print("Action: {}; by {} {:.2f} lots at {:.5f} ( ptDiff={:.1f} ({:.5f}(request.price) - {:.5f}(result.price) ))".format(
                request['type'], request['symbol'], result.volume, result.price,
                (request['price'] - result.price) * 10 ** mt5.symbol_info(request['symbol']).digits,
                request['price'], result.price)
            )
        return results

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
    utc_from = timeModel.get_utc_time_from_broker(start, timezone)
    utc_to = timeModel.get_utc_time_from_broker(end, timezone)
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