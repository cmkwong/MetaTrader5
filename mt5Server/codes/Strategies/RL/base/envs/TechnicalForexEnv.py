import numpy as np
import pandas as pd

from mt5Server.codes.Strategies.RL.base.envs.State import State
from mt5Server.codes.Backtest.func import techModel

from myUtils import dicModel


class TechnicalForexEnv:
    def __init__(self, symbol, Prices, tech_params, long_mode, all_symbols_info, time_cost_pt, commission_pt, spread_pt, random_ofs_on_reset, reset_on_close):
        self.Prices = Prices
        self.tech_params = tech_params  # pd.DataFrame
        self.dependent_datas = pd.concat([self._get_tech_df(), Prices.open, Prices.high, Prices.low, Prices.close], axis=1, join='outer').fillna(0)
        self._state = State(symbol, Prices.close, Prices.quote_exchg, self.dependent_datas, Prices.close.index,
                            time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close)
        self.random_ofs_on_reset = random_ofs_on_reset

    def _get_tech_df(self):
        tech_df = pd.DataFrame()
        for tech_name in self.tech_params.keys():
            data = techModel.get_tech_datas(self.Prices, self.tech_params[tech_name], tech_name)
            tech_df = dicModel.append_dict_df(data, tech_df, join='outer', filled=0)
        return tech_df

    def get_obs_len(self):
        obs = self.reset()
        return len(obs)

    def get_action_space_size(self):
        return self._state.action_space_size

    def reset(self):
        if not self.random_ofs_on_reset:
            self._state.reset(0)
        else:
            random_offset = np.random.randint(len(self.Prices.open) - 10)  # minus a buffer, because draw at the end of loader sometimes, then it will be bug
            self._state.reset(random_offset)
        obs = self._state.encode()
        return obs

    def step(self, action):
        reward, done = self._state.step(action)
        obs = self._state.encode()
        return obs, reward, done

