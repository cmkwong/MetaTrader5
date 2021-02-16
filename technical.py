import pandas as pd

class MovingAverage:

    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df
        self.long_mode = True

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
        # signal[0] = False   # make in count even the signal started at beginning
        # signal[-1] = False
        signal_total = 0
        if signal[0] == True:               # make in count even the signal started at beginning
            signal_total += 1
        if signal[len(signal)-1] == True:   # make in count even the signal ended at last one
            signal_total += 1
        signal_total += (signal.diff(periods=1).sum())
        signal_total /= 2
        return int(signal_total)

    def _get_signal_start_end_index(self, signal):
        """
        :param signal:
        :return:
        """
        start_index, end_index = [], []
        int_signal = signal.astype(int).diff(1)

        # buy index
        if signal[0] == True:
            start_index.append(0)
        start_index.extend([index for index in int_signal[int_signal == 1].index])

        # sell index
        end_index.extend([index for index in int_signal[int_signal == -1].index])
        if signal[len(signal) - 1] == True:
            end_index.append(len(signal)-1)

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

    def get_signal_date2(self, signal):
        """
        :param signal: Series(Boolean)
        :return: start_date_list, end_date_list
        """
        start_date_list, end_date_list = [], []
        int_signal = signal.astype(int).diff(1)

        # buy date
        if signal[0] == True:
            start_date_list.append("Before " + str(self.df['time'][0]))
        dates = list(self.df['time'][int_signal[int_signal==1].index])
        start_date_list.extend([str(date) for date in dates])

        # sell date
        dates = list(self.df['time'][int_signal[int_signal == -1].index])
        end_date_list.extend([str(date) for date in dates])
        if signal[len(signal)-1] == True:
            end_date_list.append("After " + str(self.df['time'][len(self.df)-1]))

        return start_date_list, end_date_list

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

    def _get_max_ret(self, signal):
        """
        :param signal: Series(Boolean)
        :return: float
        """
        start_index, end_index = self._get_signal_start_end_index(signal)
        ret = self._get_ret()
        rets = []
        for s,e in zip(start_index, end_index):
            rets.append(ret[s:e].sum())

    def get_ret_stat(self, signal):
        """
        :param signal: Series(Boolean)
        :return: signal_total, mean_ret, max_ret, min_ret, min_std
        """
        stat = {}
        stat["total"] = self.get_signal_total(signal)
        stat["mean"] = self.get_ret_by_signal(signal) / stat["total"]
        stat["max"] = self._get_max_ret(signal)

