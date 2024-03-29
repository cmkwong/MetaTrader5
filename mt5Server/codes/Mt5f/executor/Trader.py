from mt5Server.codes.Backtest.func import returnModel, pointsModel, timeModel
from mt5Server.codes.Mt5f.BaseMt5 import BaseMt5

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import os

class Trader(BaseMt5):
    def __init__(self, dt_string, history_path, type_filling='ioc'):
        """
        :param type_filling: 'fok', 'ioc', 'return'
        :param deviation: int
        """
        super(Trader, self).__init__()
        self.type_filling = type_filling
        self.history_path = history_path
        self.dt_string = dt_string
        self.history, self.status, self.strategy_symbols, \
        self.position_ids, self.deviations, self.avg_spreads, \
        self.open_postions, self.open_postions_date, self.close_postions, self.close_postions_date, \
        self.rets, self.earnings, self.mt5_deal_details, \
        self.q2d_at, self.open_point_diff, self.close_point_diff, self.lot_times, \
        self.long_modes = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} # see note 60b

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
        level_1_arr = np.array(['expected'] * 2 + ['real'] * 2 + ['Mt5f'] * 8 +
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
        :param requests: [request], Mt5f request: https://www.mql5.com/en/docs/constants/structures/mqltraderequest
        :param results: [result], Mt5f result: https://www.mql5.com/en/docs/constants/structures/mqltraderesult
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
        # Mt5f deal details
        self.history[strategy_id].loc[dt, ('Mt5f', 'commission')] = self.mt5_deal_details[strategy_id]['commission']
        self.history[strategy_id].loc[dt, ('Mt5f', 'swap')] = self.mt5_deal_details[strategy_id]['swap']
        self.history[strategy_id].loc[dt, ('Mt5f', 'fee')] = self.mt5_deal_details[strategy_id]['fee']
        self.history[strategy_id].loc[dt, ('Mt5f', 'earning')] = self.mt5_deal_details[strategy_id]['earning']
        self.history[strategy_id].loc[dt, ('Mt5f', 'balanced')] = self.mt5_deal_details[strategy_id]['balanced']
        self.history[strategy_id].loc[dt, ('Mt5f', 'diff')] = self.mt5_deal_details[strategy_id]['balanced'] - self.earnings[strategy_id]['expected']
        # open date and closed date
        self.history[strategy_id].loc[dt, ('Mt5f', 'open_date')] = self.open_postions_date[strategy_id]
        self.history[strategy_id].loc[dt, ('Mt5f', 'closed_date')] = self.close_postions_date[strategy_id]

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
        :param results: Mt5f results
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
        self.open_postions[strategy_id]['real'] = np.array([result.price for result in results])
        # update open pt diff
        self.open_point_diff[strategy_id] = pointsModel.get_point_diff_from_results(results, requests, expected_prices, self.all_symbol_info)
        # date
        self.open_postions_date[strategy_id] = open_position_date # update the open position time to avoid the buy again after stop loss or profit
        self.q2d_at[strategy_id] = q2d_at
        return True

    def strategy_close_update(self, strategy_id, results, requests, coefficient_vector, expected_prices, expected_ret, expected_earning, close_position_date):
        """
        :param strategy_id: str
        :param results: Mt5f results
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
        requests = self.request_format(strategy_id, lots, close_pos=False)
        spread_allowed = self.check_allowed_with_avg_spread(requests, expected_prices, self.avg_spreads[strategy_id]) # note 59a
        if not spread_allowed:
            return False, False
        results = self.request_execute(requests)

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
        requests = self.request_format(strategy_id, lots, close_pos=True)
        results = self.request_execute(requests)
        return results, requests

    def request_format(self, strategy_id, lots, close_pos=False):
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

    def request_execute(self, requests):
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