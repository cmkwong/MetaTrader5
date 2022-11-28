from mt5Server.codes.Backtest.func import pointsModel, returnModel
from mt5Server.codes.Strategies.Cointegration import coinModel

import numpy as np

class State:
    def __init__(self, symbol, close_price, quote_exchg, dependent_datas, date, time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close):
        self.symbol = symbol
        self.action_price = close_price  # close price (pd.DataFrame)
        self.quote_exchg = quote_exchg  # quote to deposit (pd.DataFrame)
        self.dependent_datas = dependent_datas  # should be shift 1 forward, because it takes action on next-day of open-price (pd.DataFrame)
        self.date = date
        self.time_cost_pt = time_cost_pt
        self.commission_pt = commission_pt
        self.spread_pt = spread_pt
        self.long_mode = long_mode
        self.all_symbols_info = all_symbols_info
        self.reset_on_close = reset_on_close
        self._init_action_space()

        self.deal_step = 0.0  # step counter from buy to sell (buy date = step 1, if sell date = 4, time cost = 3)

    def reset(self, new_offset):
        self._offset = new_offset
        self.have_position = False

    def _init_action_space(self):
        self.actions = {}
        self.actions['skip'] = 0
        self.actions['open'] = 1
        self.actions['close'] = 2
        self.action_space = list(self.actions.values())
        self.action_space_size = len(self.action_space)

    def cal_profit(self, curr_action_price, open_action_price, q2d_at):
        modified_coefficient_vector = coinModel.get_modified_coefficient_vector(np.array([]), self.long_mode, 1)  # lot_times always in 1
        return returnModel.get_value_of_earning(self.symbol, curr_action_price, open_action_price, q2d_at, self.all_symbols_info, modified_coefficient_vector)

    def encode(self):
        """
        :return: state
        """
        res = []
        earning = 0.0
        try:
            res.extend(list(self.dependent_datas.iloc[self._offset, :].values))
        except:
            print('stop')
        if self.have_position:
            earning = self.cal_profit(self.action_price.iloc[self._offset, :].values, self._prev_action_price, self.quote_exchg.iloc[self._offset, :].values)
        res.extend([earning, float(self.have_position)])  # earning, have_position (True = 1.0, False = 0.0)
        return np.array(res, dtype=np.float32)

    def step(self, action):
        """
        Calculate the rewards and check if the env is done
        :param action: long/short * Open/Close/hold position: 6 actions
        :return: reward, done
        """
        done = False
        reward = 0.0  # in deposit USD
        curr_action_price = self.action_price.iloc[self._offset].values[0]
        q2d_at = self.quote_exchg.iloc[self._offset].values[0]

        if action == self.actions['open'] and not self.have_position:
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.spread_pt, q2d_at, self.all_symbols_info)  # spread cost
            self.openPos_price = curr_action_price
            self.have_position = True

        elif action == self.actions['close'] and self.have_position:
            reward += self.cal_profit(curr_action_price, self._prev_action_price, q2d_at)  # calculate the profit
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.time_cost_pt, q2d_at, self.all_symbols_info)  # time cost
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.spread_pt, q2d_at, self.all_symbols_info)  # spread cost
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.commission_pt, q2d_at, self.all_symbols_info)  # commission cost
            self.have_position = False
            if self.reset_on_close:
                done = True

        elif action == self.actions['skip'] and self.have_position:
            reward += self.cal_profit(curr_action_price, self._prev_action_price, q2d_at)
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.time_cost_pt, q2d_at, self.all_symbols_info)  # time cost
            self.deal_step += 1

        # update status
        self._prev_action_price = curr_action_price
        self._offset += 1
        if self._offset >= len(self.action_price) - 1:
            done = True

        return reward, done