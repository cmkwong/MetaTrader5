import talib
from production.codes.lib.BaseTechical import BaseTechical
from production.codes.lib.Trader import Data
from production.codes.lib.config import *

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
class MACDTechnical(BaseTechical):
    def __init__(self, df, long_mode):
        super(MACDTechnical, self).__init__(df)
        self._df = df
        self.long_mode = long_mode

    def _get_macd(self, fastperiod, slowperiod, signalperiod):
        macd = talib.MACD(self._df['close'], fastperiod, slowperiod, signalperiod)
        return macd

    def get_signal(self, fastperiod=12, slowperiod=26, signalperiod=9):
        """
        :param period: int
        :param th: int: 0-100(+/-ve)
        :param upper: Boolean
        :return:
        """
        macd, macdsignal, macdhist = self._get_macd(fastperiod, slowperiod, signalperiod)
        if self.long_mode:
            signal = macd > 0
        else:
            signal = macd < 0
        return signal

df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)
macdTechnical = MACDTechnical(df, long_mode=True)
signal = macdTechnical.get_signal(fastperiod=10, slowperiod=100, signalperiod=9)