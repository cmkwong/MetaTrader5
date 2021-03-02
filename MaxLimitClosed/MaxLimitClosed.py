import pandas as pd
import numpy as np
from production.codes.common.BaseTechical import BaseTechical

class MaxLimitClosed(BaseTechical):

    def __init__(self, limit_unit):
        super(MaxLimitClosed, self).__init__()
        self.limit_unit = limit_unit # if limit_unit = 0, that is disable

    def simple_limit_end_index(self, starts, ends):
        """
        modify the ends_index, eg. close the trade until specific unit
        :param starts: list [int] index
        :param ends: list [int] index
        :return: starts, ends
        """
        new_starts, new_ends = [], []
        for s, e in zip(starts, ends):
            new_starts.append(s)
            new_end = min(s + self.limit_unit, e)
            new_ends.append(new_end)
        return new_starts, new_ends

    def modify(self, signal):
        """
        :param signal(backtesting): Series [Boolean]
        :return: modified_signal: Series
        """
        assert signal[0] != True, "Signal not for backtesting"
        assert signal[len(signal) - 1] != True, "Signal not for backtesting"
        assert signal[len(signal) - 2] != True, "Signal not for backtesting"

        if self.limit_unit > 0:
            int_signal = self._get_int_signal(signal)
            signal_starts = [i - 1 for i in self._get_open_index(int_signal)]
            signal_ends = [i - 1 for i in self._get_close_index(int_signal)]
            starts, ends = self.simple_limit_end_index(signal_starts, signal_ends)

            # assign new signal
            signal[:] = False
            for s, e in zip(starts, ends):
                signal[s:e] = True
        return signal





