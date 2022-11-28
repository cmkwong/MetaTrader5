import sys

sys.path.append("C:/Users/Chris/projects/210215_mt5/mt5Server")
sys.path.append("C:/Users/Chris/projects/AtomLib")

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping


class Backtest_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, startTime, endTime, lot=1):
        super(Backtest_SwingScalping, self).__init__(mt5Controller, symbol)
        self.startTime = startTime
        self.endTime = endTime
        self.LOT = lot
        self.prepare1MinData(startTime, endTime)

    def run(self, diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
            lowerEma=25, middleEma=50, upperEma=100):
        fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

        masterSignal = self.getMasterSignal(fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)

        return masterSignal


mt5Controller = MT5Controller()
backtest_SwingScalping = Backtest_SwingScalping(mt5Controller, 'USDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
masterSignal = backtest_SwingScalping.run(diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
                           lowerEma=25, middleEma=50, upperEma=100)
