import talib
from production.codes.lib.BaseTechical import BaseTechical
from production.codes.lib.Trader import Data
from production.codes.lib.config import *

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
class RSITechnical(BaseTechical):
    def __int__(self, df):
        super(RSITechnical, self).__init__(df)
        self._df = df

    def _get_rsi(self, period):
        rsi = talib.RSI(self._df['close'], timeperiod=period)
        return rsi

    def get_signal(self, period, th):
        """
        :param period: int
        :param th: int: 0-100(+/-ve)
        :param upper: Boolean
        :return:
        """
        rsi = self._get_rsi(period)
        if th > 0:
            signal = rsi >= abs(th)
        else:
            signal = rsi <= abs(th)
        return signal

df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)
rsiTechnical = RSITechnical(df)
rsi = rsiTechnical._get_rsi(period=14)
signal = rsiTechnical.get_signal(period=14, th=-30)

print()