import sys
sys.path.append("C:/Users/Chris/projects/210215_mt5/mt5Server")
sys.path.append("C:/Users/Chris/projects/AtomLib")

from mt5Server.codes.Backtest.func import timeModel
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping

import os
import csv
import numpy as np
import time

class Train_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, startTime, endTime, lot=1):
        super(Train_SwingScalping, self).__init__(mt5Controller, symbol)
        self.startTime = startTime
        self.endTime = endTime
        self.LOT = lot
        self.prepare1MinData(startTime, endTime)

    # def test(self, diff_ema_upper_middle=20, diff_ema_middle_lower=20, ratio_sl_sp=1.2,
    #          lowerEma=92, middleEma=95, upperEma=98):
    #     ohlcvs = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')
    #
    #     # get the master signal
    #     masterSignal = self.getMasterSignal(ohlcvs,
    #                                         lowerEma, middleEma, upperEma,
    #                                         diff_ema_upper_middle, diff_ema_middle_lower,
    #                                         ratio_sl_sp)
    #
    #     return masterSignal

    # calculate the win rate
    def getWinRate(self, masterSignal, trendType='rise'):
        count = (masterSignal['earning_' + trendType] != 0).sum()
        positiveProfit = (masterSignal['earning_' + trendType] > 0).sum()
        if count == 0:
            winRate = 0.0
        else:
            winRate = "{:.2f},".format((positiveProfit / count) * 100)
        return count, winRate

    # calculate the profit
    def getProfit(self, masterSignal, trendType='rise'):
        return "{:.2f}".format(masterSignal['earning_' + trendType].sum())

    def getSummary(self, masterSignal, trendType='rise', *params):
        count, winRate = self.getWinRate(masterSignal, trendType)
        profit = self.getProfit(masterSignal, trendType)
        summary = {'type': trendType,
                   'count': count,
                   'winRate': winRate,
                   'profit': profit,
                   'ratio_sl_sp': params[0],
                   'diff_ema_middle_lower': params[1],
                   'diff_ema_upper_middle': params[2],
                   'upperEma': params[3],
                   'middleEma': params[4],
                   'lowerEma': params[5]
                   }
        return summary

    def loopRun(self):
        # define the writer
        r = 0
        # fetch data from database
        fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

        for ratio_sl_sp in np.arange(1.2, 2.2, 0.2):
            for diff_ema_middle_lower in np.arange(20, 80, 10):
                for diff_ema_upper_middle in np.arange(20, 80, 10):
                    # if diff_ema_upper_middle <= 50:
                    #     print('continue middle')
                    #     continue
                    for upperEma in reversed(np.arange(20, 100, 4)):
                        for middleEma in reversed(np.arange(19, upperEma - 1, 4)):
                            for lowerEma in reversed(np.arange(18, middleEma - 1, 4)):
                                # getting master signal
                                start = time.time()
                                masterSignal = self.getMasterSignal(fetchData_cust,
                                                                    lowerEma, middleEma, upperEma,
                                                                    diff_ema_upper_middle, diff_ema_middle_lower,
                                                                    ratio_sl_sp)

                                # build the dictionary to write into csv
                                riseSummary = self.getSummary(masterSignal, 'rise', ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma)
                                downSummary = self.getSummary(masterSignal, 'down', ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma)

                                with open(os.path.join(self.backTestDocPath, self.baclTestDocName), 'a', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    # write header
                                    if r == 0:
                                        writer.writerow(riseSummary.keys())
                                    # write the rows
                                    writer.writerow(riseSummary.values())
                                    writer.writerow(downSummary.values())
                                    print(riseSummary)
                                    print(downSummary)
                                    r += 2
                                processTime = time.time() - start
                                print(f"Overall Process Time: {processTime}")


sybmols = ['GBPUSD', 'CADJPY', 'AUDJPY', 'AUDUSD', 'USDCAD', 'USDJPY', 'EURCAD', 'EURUSD']
mt5Controller = MT5Controller()
backTest_SwingScalping = Train_SwingScalping(mt5Controller, 'EURUSD', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
# backTest_SwingScalping.test()
backTest_SwingScalping.loopRun()
