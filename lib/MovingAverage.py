from production.codes.lib.BaseTechical import BaseTechical

class MovingAverage(BaseTechical):

    def __init__(self, df, backtest=True, long_mode=True):
        super(MovingAverage, self).__init__(df)
        self._df = df
        self._backtest = backtest # if backtesting, discard redundant signal (head and tail)
        self.long_mode = long_mode

    def get_moving_average(self, m):
        """
        :param m: int
        :return: Series
        """
        series = self._df['close'].rolling(m).sum()/m
        return series

    def get_signal(self, slow, fast, limit_unit):
        """
        :param slow: Series
        :param fast: Series
        :return: Series(Boolean)
        """
        if self.long_mode:
            signal = fast > slow
        else:
            signal = fast < slow
        if self._backtest: # # discard if had ahead signal or tailed signal
            signal = self._discard_head_signal(signal)
            signal = self._discard_tail_signal(signal)
        if limit_unit > 0:
            signal = self._maxLimitClosed(signal, limit_unit)
        return signal

    # def get_ret_stat(self, signal, slow_index, fast_index):
    #     """
    #     :param signal: Series(Boolean)
    #     :return: signal_total, mean_ret, max_ret, min_ret, min_std
    #     """
    #     stat = {}
    #     if signal.sum() != 0:
    #         stat["slow"] = slow_index
    #         stat["fast"] = fast_index
    #         stat["count"] = self._get_action_total(signal)
    #         stat["accum"] = self._get_accum_ret(signal)
    #         stat["mean"] = np.mean(self.get_ret_list(signal))
    #         stat["max"] = np.max(self.get_ret_list(signal))
    #         stat["min"] = np.min(self.get_ret_list(signal))
    #         stat["std"] = np.std(self.get_ret_list(signal))
    #         stat["acc"] = self._get_accuracy(signal)
    #         stat["limit"] = self.limit_unit
    #     return stat

