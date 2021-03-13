import numpy as np

class BaseTechical:

    def __init__(self, df=None):
        self._df = df

    def _discard_head_signal(self, signal):
        """
        :param signal: Series
        :return: signal: Series
        """
        if signal[0] == True:
            for index, value in signal.items():
                if value == True:
                    signal[index] = False
                else:
                    break
        return signal

    def _discard_tail_signal(self, signal):
        """
        :param signal: Series
        :return: signal: Series
        """
        if signal[len(signal ) -1] == True or signal[len(signal ) -2] == True: # See Note 6. and 11.
            length = len(signal)
            signal[length -1] = True # Set the last index is True, it will set back to false in following looping
            for ii, value in enumerate(reversed(signal.values)):
                if value == True:
                    signal[length - 1 - ii] = False
                else:
                    break
        return signal

    def _get_int_signal(self, signal):
        int_signal = signal.astype(int).diff(1)
        return int_signal

    def _get_open_index(self, int_signal):
        start_index = []
        start_index.extend \
            ([index + 1 for index in int_signal[int_signal == 1].index])  # see note point 6 why added by 1
        return start_index

    def _get_close_index(self, int_signal):
        end_index = []
        end_index.extend([index + 1 for index in int_signal[int_signal == -1].index]) # see note point 6 why added by 1
        return end_index

    def _get_action_start_end_index(self, signal):
        """
        :param signal: Series
        :return: list: start_index, end_index
        """
        int_signal = self._get_int_signal(signal)

        # buy index
        start_index = self._get_open_index(int_signal)

        # sell index
        end_index = self._get_close_index(int_signal)

        # # modify the start_index, end_index, if needed
        # if self.limit_unit > 0:
        #     start_index, end_index = self._simple_limit_end_index(start_index, end_index)

        return start_index, end_index

    def _get_change(self):
        """
        :return: change: Series
        """
        diffs = self._df['open'].diff(periods=1)
        shifts = self._df['open'].shift(1)
        change = diffs / shifts
        return change

    def _get_ret(self):
        """
        :return: ret: Series
        """
        change = self._get_change()
        ret = 1 + change
        return ret

    def _get_accuracy(self, signal):
        ret_list = self.get_ret_list(signal)
        accuracy = np.sum([r > 1 for r in ret_list]) / len(ret_list)
        return accuracy

    def _get_accum_ret(self, signal):
        """
        :param signal: Series(Boolean)
        :return: ret_by_signal: float64
        """
        ret_by_signal = 1
        ret_list = self.get_ret_list(signal)
        for ret in ret_list:
            ret_by_signal *= ret
        return ret_by_signal

    def _get_action_total(self, signal):
        """
        :param signal: Series(Boolean)
        :return: int
        """
        start, end = self._get_action_start_end_index(signal)

        return len(start)

    def _get_action_date(self, signal):
        """
        :param signal: Series(Boolean)
        :return: start_date_list, end_date_list
        """
        start_date_list, end_date_list = [], []
        int_signal = signal.astype(int).diff(1)
        start_index, end_index = self._get_action_start_end_index(signal)
        # buy date
        dates = list(self._df['time'][start_index])
        start_date_list.extend([str(date) for date in dates])

        # sell date
        dates = list(self._df['time'][end_index])
        end_date_list.extend([str(date) for date in dates])

        return start_date_list, end_date_list

    def get_ret_list(self, signal):
        """
        :param signal: Series(Boolean)
        :return: float
        """
        start_index, end_index = self._get_action_start_end_index(signal)
        ret = self._get_ret()
        rets = []
        for s ,e in zip(start_index, end_index):
            rets.append(ret[ s +1: e +1].prod()) # see notes point 6
        return rets

    def get_action_detail(self, signal):
        """
        :param signal: Series
        :return: details: dictionary
        """
        details = {}
        start_dates, end_dates = self._get_action_date(signal)
        ret_list = self.get_ret_list(signal)
        for s, e, r in zip(start_dates, end_dates, ret_list):
            key = s + '-' + e
            details[key] = r
        return details

    def _simple_limit_end_index(self, starts, ends, limit_unit):
        """
        modify the ends_index, eg. close the trade until specific unit
        :param starts: list [int] index
        :param ends: list [int] index
        :return: starts, ends
        """
        new_starts, new_ends = [], []
        for s, e in zip(starts, ends):
            new_starts.append(s)
            new_end = min(s + limit_unit, e)
            new_ends.append(new_end)
        return new_starts, new_ends

    def _maxLimitClosed(self, signal, limit_unit):
        """
        :param signal(backtesting): Series [Boolean]
        :return: modified_signal: Series
        """
        assert signal[0] != True, "Signal not for backtesting"
        assert signal[len(signal) - 1] != True, "Signal not for backtesting"
        assert signal[len(signal) - 2] != True, "Signal not for backtesting"

        int_signal = self._get_int_signal(signal)
        signal_starts = [i - 1 for i in self._get_open_index(int_signal)]
        signal_ends = [i - 1 for i in self._get_close_index(int_signal)]
        starts, ends = self._simple_limit_end_index(signal_starts, signal_ends, limit_unit)

        # assign new signal
        signal[:] = False
        for s, e in zip(starts, ends):
            signal[s:e] = True
        return signal

    def get_stat(self, signal):
        """
        :return: stat dictionary
        """
        stat = {}
        if signal.sum() != 0:
            stat["count"] = self._get_action_total(signal)
            stat["accum"] = self._get_accum_ret(signal)
            stat["mean"] = np.mean(self.get_ret_list(signal))
            stat["max"] = np.max(self.get_ret_list(signal))
            stat["min"] = np.min(self.get_ret_list(signal))
            stat["std"] = np.std(self.get_ret_list(signal))
            stat["acc"] = self._get_accuracy(signal)
        return stat