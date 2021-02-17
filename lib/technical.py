import pandas as pd
import numpy as np

class MovingAverage:

    def __init__(self, symbol, df, long_mode=True):
        self.symbol = symbol
        self.df = df
        self.long_mode = long_mode

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

    def get_signal_total(self, signal):
        """
        :param signal: Series(Boolean)
        :return: int
        """
        start, end = self._get_signal_start_end_index(signal)

        return len(start)

    def _get_signal_start_end_index(self, signal):
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
        if signal[len(signal)-1] == True:
            discard_last_buy_index = True

        # buy index
        start_index.extend([index for index in int_signal[int_signal == 1].index])
        if discard_last_buy_index:
            start_index.pop(-1)

        # sell index
        end_index.extend([index for index in int_signal[int_signal == -1].index])
        if discard_first_sell_index:
            end_index.pop(0)

        return start_index, end_index

    def get_signal_date(self, signal):
        """
        :param signal: Series(Boolean)
        :return: start_date_list, end_date_list
        """
        start_date_list, end_date_list = [], []
        int_signal = signal.astype(int).diff(1)
        start_index, end_index = self._get_signal_start_end_index(signal)
        # buy date
        dates = list(self.df['time'][start_index])
        start_date_list.extend([str(date) for date in dates])

        # sell date
        dates = list(self.df['time'][end_index])
        end_date_list.extend([str(date) for date in dates])

        return start_date_list, end_date_list

    def get_signal_detail(self, signal):
        """
        :param signal: Series
        :return: details: dictionary
        """
        details = {}
        start_dates, end_dates = self.get_signal_date(signal)
        ret_list = self.get_ret_list(signal)
        for s, e, r in zip(start_dates, end_dates, ret_list):
            key = s + '-' + e
            details[key] = r
        return details

    def _get_ret(self):
        """
        :param signal: Series(Boolean)
        :return: ret: Series
        """
        diffs = self.df['close'].diff(periods=1)
        shifts = self.df['close'].shift(1)
        ret = diffs / shifts
        return ret

    def get_ret_by_signal(self, signal):
        """
        :param signal: Series(Boolean)
        :return: ret_by_signal: float64
        """
        ret = self._get_ret()
        ret_by_signal = (ret * signal).sum()
        return ret_by_signal

    def get_ret_list(self, signal):
        """
        :param signal: Series(Boolean)
        :return: float
        """
        start_index, end_index = self._get_signal_start_end_index(signal)
        ret = self._get_ret()
        rets = []
        for s,e in zip(start_index, end_index):
            rets.append(ret[s:e].sum())
        return rets

    def _get_accuracy(self, signal):
        ret_list = self.get_ret_list(signal)
        accuracy = np.sum([i > 0 for i in ret_list]) / len(ret_list)
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
            stat["count"] = self.get_signal_total(signal)
            stat["accum"] = self.get_ret_by_signal(signal)
            stat["mean"] = np.mean(self.get_ret_list(signal))
            stat["max"] = np.max(self.get_ret_list(signal))
            stat["min"] = np.min(self.get_ret_list(signal))
            stat["std"] = np.std(self.get_ret_list(signal))
            stat["acc"] = self._get_accuracy(signal)
        return stat

