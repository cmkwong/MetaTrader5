from mt5Server.codes.Backtest.func import timeModel
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping

import os
import csv
import numpy as np

class BackTest_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, startTime, endTime, breakThroughCondition='50', lot=1):
        super(BackTest_SwingScalping, self).__init__(mt5Controller, symbol)
        self.startTime = startTime
        self.endTime = endTime
        self.breakThroughCondition = breakThroughCondition
        self.LOT = lot
        self.prepare1MinData(startTime, endTime)

    def test(self, diff_ema_upper_middle=50, diff_ema_middle_lower=50, ratio_sl_sp=1.5,
             lowerEma=25, middleEma=50, upperEma=100):
        fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

        # get the master signal
        masterSignal = self.getMasterSignal(fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)

        return masterSignal

    def getHeader(self):
        header = ['type', 'count', 'winRate', 'profit', 'ratio_sl_sp', 'diff_ema_middle_lower', 'diff_ema_upper_middle', 'upperEma', 'middleEma', 'lowerEma']
        txt = ','.join(header)
        return txt

    def writeRowSummaryTxt(self, masterSignal, trendType ='rise', *params):
        # rise type
        rowTxt = "{},".format(trendType)
        # total trade
        total = (masterSignal['earning_' + trendType] != 0).sum()
        rowTxt += "{},".format(total)
        if total == 0:
            rowTxt += '\n'
            return rowTxt
        # winRate
        positiveProfit = (masterSignal['earning_' + trendType] > 0).sum()
        rowTxt += "{:.2f},".format((positiveProfit / total) * 100)
        # profit
        rowTxt += "{:.2f},".format(masterSignal['earning_' + trendType].sum())
        # ratio_sl_sp
        rowTxt += "{},".format(params[0])
        # diff_ema_middle_lower
        rowTxt += "{},".format(params[1])
        # diff_ema_upper_middle
        rowTxt += "{},".format(params[2])
        # upperEma
        rowTxt += "{},".format(params[3])
        # middleEma
        rowTxt += "{},".format(params[4])
        # lowerEma
        rowTxt += "{}\n".format(params[5])
        return rowTxt

    def writeRowSummary(self, masterSignal, trendType = 'rise', *params):
        rowList = []
        # rise type
        rowList.append(trendType)
        # total trade
        total = (masterSignal['earning_' + trendType] != 0).sum()
        rowList.append(total)
        if total == 0:
            return rowList
        # winRate
        positiveProfit = (masterSignal['earning_' + trendType] > 0).sum()
        rowList.append("{:.2f}".format((positiveProfit / total) * 100))
        # profit
        rowList.append("{:.2f}".format(masterSignal['earning_' + trendType].sum()))
        # ratio_sl_sp
        rowList.append("{}".format(params[0]))
        # diff_ema_middle_lower
        rowList.append("{}".format(params[1]))
        # diff_ema_upper_middle
        rowList.append("{}".format(params[2]))
        # upperEma
        rowList.append("{}".format(params[3]))
        # middleEma
        rowList.append("{}".format(params[4]))
        # lowerEma
        rowList.append("{}".format(params[5]))
        return rowList

    def loopRun(self):
        with open(os.path.join(self.backTestDocPath, self.baclTestDocName), 'w') as f:
            # define the writer
            writer = csv.writer(f, delimiter=",")
            # write header
            writer.writerow(['type', 'count'])

            # fetch data from database
            fetchData_cust = self.dataFeeder.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

            for ratio_sl_sp in np.arange(1.2, 2.0, 0.1):
                for diff_ema_middle_lower in np.arange(20, 80, 2):
                    for diff_ema_upper_middle in np.arange(20, 80, 2):
                        for upperEma in reversed(np.arange(20, 100, 2)):
                            for middleEma in reversed(np.arange(19, 99, 2)):
                                for lowerEma in reversed(np.arange(18, 98, 2)):

                                    # getting master signal
                                    masterSignal = self.getMasterSignal(fetchData_cust,
                                                         lowerEma, middleEma, upperEma,
                                                         diff_ema_upper_middle, diff_ema_middle_lower,
                                                         ratio_sl_sp)
                                    # write the rows
                                    writer.writerow(self.writeRowSummary(masterSignal, 'rise', ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma))
                                    writer.writerow(self.writeRowSummary(masterSignal, 'down', ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma))


sybmols = ['GBPUSD', 'CADJPY', 'AUDJPY', 'AUDUSD', 'USDCAD', 'USDJPY', 'EURCAD', 'EURUSD']
mT5Controller = MT5Controller()
backTest_SwingScalping = BackTest_SwingScalping(mT5Controller, 'USDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
# backTest_SwingScalping.test()
backTest_SwingScalping.loopRun()
