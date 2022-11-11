from mt5Server.codes.Backtest.func import timeModel
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Data.DataFeeder import DataFeeder
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping

import numpy as np

class BackTest_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, startTime, endTime, breakThroughCondition='50', lot=1):
        super(BackTest_SwingScalping, self).__init__(mt5Controller, symbol)
        self.dataFeeder = DataFeeder(mt5Controller)
        self.startTime = startTime
        self.endTime = endTime
        self.breakThroughCondition = breakThroughCondition
        self.LOT = lot

    def run(self, diff_ema_upper_middle=50, diff_ema_middle_lower=50, ratio_sl_sp=1.5,
            lowerEma=25, middleEma=50, upperEma=100):
        fetchData_min = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='1min')
        fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')
        fetchData_cust['lower'] = techModel.get_EMA(fetchData_cust.close, lowerEma)
        fetchData_cust['middle'] = techModel.get_EMA(fetchData_cust.close, middleEma)
        fetchData_cust['upper'] = techModel.get_EMA(fetchData_cust.close, upperEma)

        # calculate the points difference
        fetchData_cust['ptDiff_100_50'], fetchData_cust['ptDiff_50_25'] = self.getPointDiff(self.symbol, fetchData_cust['upper'], fetchData_cust['middle'], fetchData_cust['lower'])

        fetchData_cust['riseBreak'], fetchData_cust['downBreak'] = self.getBreakThroughSignal(fetchData_cust.loc[:,('open', 'high', 'low', 'close')], fetchData_cust.loc[:, ('lower', 'middle', 'upper')])

        # get the signals
        fetchData_cust['riseDiff'] = (fetchData_cust['ptDiff_100_50'] <= -diff_ema_upper_middle) & (fetchData_cust['ptDiff_50_25'] <= -diff_ema_middle_lower)
        fetchData_cust['downDiff'] = (fetchData_cust['ptDiff_100_50'] >= diff_ema_upper_middle) & (fetchData_cust['ptDiff_50_25'] >= diff_ema_middle_lower)

        masterSignal = self.getMasterSignal(fetchData_cust.loc[:, ('open', 'high', 'low', 'close')],
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)

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