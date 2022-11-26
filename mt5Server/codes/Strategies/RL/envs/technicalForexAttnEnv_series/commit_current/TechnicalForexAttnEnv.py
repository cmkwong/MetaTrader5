import numpy as np

from RL.envs.TechnicalForexEnv import TechnicalForexEnv
from RL.envs.attnState_series.commit_current.AttnState import AttnState


class TechnicalForexAttnEnv(TechnicalForexEnv):
    def __init__(self, seqLen, symbol, Prices, tech_params, long_mode, all_symbols_info, time_cost_pt, commission_pt, spread_pt, random_ofs_on_reset, reset_on_close):
        super(TechnicalForexAttnEnv, self).__init__(symbol, Prices, tech_params, long_mode, all_symbols_info, time_cost_pt, commission_pt, spread_pt, random_ofs_on_reset, reset_on_close)
        self.seqLen = seqLen
        self._state = AttnState(seqLen, symbol, Prices.c, Prices.quote_exchg, self.dependent_datas, Prices.c.index,
                                time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close)

    def reset(self):
        startIndex = len(self.Prices.c) % self.seqLen + self.seqLen * 1 # (+ self.seqLen * 1) because of it taking backward seq as input
        if not self.random_ofs_on_reset:
            self._state.reset(startIndex)
        else:
            random_offset = np.random.randint(startIndex, (len(self.Prices.c) - startIndex) / self.seqLen)  # minus a buffer, because draw at the end of loader sometimes, then it will be bug
            self._state.reset(random_offset * self.seqLen)
        obs = self._state.encode()
        return obs