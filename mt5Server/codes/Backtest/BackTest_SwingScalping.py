from mt5Server.codes.Backtest.func import timeModel
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping

import numpy as np

class BackTest_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, startTime, endTime, breakThroughCondition='50', lot=1):
        super(BackTest_SwingScalping, self).__init__(mt5Controller, symbol)
        self.startTime = startTime
        self.endTime = endTime
        self.breakThroughCondition = breakThroughCondition
        self.LOT = lot
        self.prepare1MinData(startTime, endTime)

    def run(self, diff_ema_upper_middle=50, diff_ema_middle_lower=50, ratio_sl_sp=1.5,
            lowerEma=25, middleEma=50, upperEma=100):
        fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

        # get the master signal
        masterSignal = self.getMasterSignal(fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)

        masterSignal['earning_rise'] = masterSignal.apply(lambda r: self.getEarning(r.name, r['riseBreak'], r['riseRange'], r['open'], r['quote_exchg'], r['stopLoss'], r['takeProfit'], 'rise'), axis=1)
        masterSignal['earning_down'] = masterSignal.apply(lambda r: self.getEarning(r.name, r['downBreak'], r['downRange'], r['open'], r['quote_exchg'], r['stopLoss'], r['takeProfit'], 'down'), axis=1)
        print()

    def _backup(self):
        for diff_ema_50_25 in np.arange(20, 80, 2):
            for diff_ema_100_50 in np.arange(20, 80, 2):
                for ema100Percent in reversed(np.arange(20, 100, 2)):
                    for ema50Percent in reversed(np.arange(19, 99, 2)):
                        for ema25Percent in reversed(np.arange(18, 98, 2)):
                            pass

mT5Controller = MT5Controller()
backTest_SwingScalping = BackTest_SwingScalping(mT5Controller, 'USDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
backTest_SwingScalping.run()