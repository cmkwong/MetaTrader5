import numpy as np

from RL.envs.State import State


class AttnState(State):
    def __init__(self, seqLen, symbol, close_price, quote_exchg, dependent_datas, date, time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close):
        super(AttnState, self).__init__(symbol, close_price, quote_exchg, dependent_datas, date, time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close)
        self.seqLen = seqLen

    def encode(self):
        """
        :return: state
        """
        state = {}
        earning = 0.0
        if self.have_position:
            earning = self.cal_profit(self.action_price.iloc[self._offset, :].values, self._prev_action_price, self.quote_exchg.iloc[self._offset, :].values)
        state['encoderInput'] = self.dependent_datas.iloc[self._offset - self.seqLen:self._offset, :].values  # getting seqLen * 2 len of data
        state['status'] = np.array([earning, float(self.have_position)])  # earning, have_position (True = 1.0, False = 0.0)
        return state
