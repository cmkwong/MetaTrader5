import pandas as pd
import numpy as np

class MovingAverage:

    def __init__(self, symbol, df, long_mode=True, limit_unit=0):
        self.symbol = symbol
        self.df = df
        self.long_mode = long_mode
        self.limit_unit = limit_unit     # if larger than 0, end_index will be limited

    def get_moving_average(self, m):
        """
        :param m: int
        :return: Series
        """
        series = self.df['close'].rolling(m).sum()/m
        return series

    def get_signal(self, slow, fast):
        """
        :param slow: Series
        :param fast: Series
        :return: Series(Boolean)
        """
        if self.long_mode:
            signal = fast > slow
        else:
            signal = fast < slow
        return signal

    def get_action_total(self, signal):
        """
        :param signal: Series(Boolean)
        :return: int
        """
        start, end = self._get_action_start_end_index(signal)

        return len(start)

    def _get_action_start_end_index(self, signal):
        """
        :param signal: Series
        :return: list: start_index, end_index
        """
        start_index, end_index = [], []
        discard_first_sell_index, discard_last_buy_index = False, False
        int_signal = signal.astype(int).diff(1)

        # discard if had ahead signal or tailed signal
        if signal[0] == True:
            discard_first_sell_index = True
        if signal[len(signal)-1] == True or signal[len(signal)-2] == True: # See Note point 6
            discard_last_buy_index = True

        # buy index
        start_index.extend([index+1 for index in int_signal[int_signal == 1].index]) # see note point 6 why added by 1
        if discard_last_buy_index:
            start_index.pop(-1)

        # sell index
        end_index.extend([index+1 for index in int_signal[int_signal == -1].index])
        if discard_first_sell_index:
            end_index.pop(0)

        # modify the start_index, end_index, if needed
        if self.limit_unit > 0:
            start_index, end_index = self._simple_limit_end_index(start_index, end_index)

        return start_index, end_index

    def _simple_limit_end_index(self, starts, ends):
        """
        modify the ends_index, eg. close the trade until specific unit
        :param starts: list [int]
        :param ends: list [int]
        :return: starts, ends
        """
        new_starts, new_ends = [], []
        for s, e in zip(starts, ends):
            new_starts.append(s)
            new_end = min(s + self.limit_unit, e)
            new_ends.append(new_end)
        return new_starts, new_ends

    def get_action_date(self, signal):
        """
        :param signal: Series(Boolean)
        :return: start_date_list, end_date_list
        """
        start_date_list, end_date_list = [], []
        int_signal = signal.astype(int).diff(1)
        start_index, end_index = self._get_action_start_end_index(signal)
        # buy date
        dates = list(self.df['time'][start_index])
        start_date_list.extend([str(date) for date in dates])

        # sell date
        dates = list(self.df['time'][end_index])
        end_date_list.extend([str(date) for date in dates])

        return start_date_list, end_date_list

    def get_action_detail(self, signal):
        """
        :param signal: Series
        :return: details: dictionary
        """
        details = {}
        start_dates, end_dates = self.get_action_date(signal)
        ret_list = self.get_ret_list(signal)
        for s, e, r in zip(start_dates, end_dates, ret_list):
            key = s + '-' + e
            details[key] = r
        return details

    def _get_change(self):
        """
        :return: change: Series
        """
        diffs = self.df['open'].diff(periods=1)
        shifts = self.df['open'].shift(1)
        change = diffs / shifts
        return change

    def _get_ret(self):
        """
        :return: ret: Series
        """
        change = self._get_change()
        ret = 1 + change
        return ret

    def get_accum_ret(self, signal):
        """
        :param signal: Series(Boolean)
        :return: ret_by_signal: float64
        """
        ret_by_signal = 1
        ret_list = self.get_ret_list(signal)
        for ret in ret_list:
            ret_by_signal *= ret
        return ret_by_signal

    def get_ret_list(self, signal):
        """
        :param signal: Series(Boolean)
        :return: float
        """
        start_index, end_index = self._get_action_start_end_index(signal)
        ret = self._get_ret()
        rets = []
        for s,e in zip(start_index, end_index):
            rets.append(ret[s+1:e+1].prod()) # see notes point 6
        return rets

    def _get_accuracy(self, signal):
        ret_list = self.get_ret_list(signal)
        accuracy = np.sum([r > 1 for r in ret_list]) / len(ret_list)
        return accuracy

    def get_ret_stat(self, signal, slow_index, fast_index):
        """
        :param signal: Series(Boolean)
        :return: signal_total, mean_ret, max_ret, min_ret, min_std
        """
        stat = {}
        if signal.sum() != 0:
            stat["slow"] = slow_index
            stat["fast"] = fast_index
            stat["count"] = self.get_action_total(signal)
            stat["accum"] = self.get_accum_ret(signal)
            stat["mean"] = np.mean(self.get_ret_list(signal))
            stat["max"] = np.max(self.get_ret_list(signal))
            stat["min"] = np.min(self.get_ret_list(signal))
            stat["std"] = np.std(self.get_ret_list(signal))
            stat["acc"] = self._get_accuracy(signal)
        return stat

