import sys
import os

sys.path.append('C:/Users/Chris/projects/210215_mt5')
sys.path.append('C:/Users/Chris/projects/AtomLib')
from myBacktest import techModel
from mt5Server.codes.Strategies.SwingScalping.Base_SwingScalping import Base_SwingScalping
from myUtils.printModel import print_at

import pandas as pd
import collections
import time


class Live_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, *,
                 diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
                 lowerEma=25, middleEma=50, upperEma=100,
                 trendType='rise',
                 lot=1,
                 auto=False, tg=None):
        super(Live_SwingScalping, self).__init__(mt5Controller, symbol)
        self.diff_ema_upper_middle = diff_ema_upper_middle
        self.diff_ema_middle_lower = diff_ema_middle_lower
        self.ratio_sl_sp = ratio_sl_sp
        self.lowerEma = lowerEma
        self.middleEma = middleEma
        self.upperEma = upperEma

        self.trendType = trendType  # 'rise' / 'down'

        # lot
        self.LOT = lot
        # init the variables
        self.breakThroughTime = None
        self.breakThroughCondition, self.trendRangeCondition = False, False
        # check if notice have been announced
        self.inPosition = False
        # auto
        self.auto = auto
        # define tg
        self.tg = tg

    @property
    def getName(self):
        return f"{self.__class__.__name__}_{self.symbol}: {self.trendType} {self.diff_ema_upper_middle} {self.diff_ema_middle_lower} {self.lowerEma} {self.middleEma} {self.upperEma} {self.ratio_sl_sp} {self.LOT}"

    def checkValidAction(self, masterSignal, trendType='rise'):
        lastRow = masterSignal.iloc[-1, :]
        currentIndex = masterSignal.index[-1]
        # meet the conditions and have not trade that before and not in-position
        if (lastRow[trendType + 'Break'] and lastRow[trendType + 'Range'] and currentIndex != self.breakThroughTime and not self.inPosition):
            print(f'{self.symbol} {self.trendType} action going ... ')
            self.status = {'type': trendType, 'time': self.breakThroughTime, 'sl': lastRow['stopLoss'], 'tp': lastRow['takeProfit']}
            print(self.status)
            # make notice and take action if auto set to True
            statusTxt = f'{self.symbol}\n'
            for k, v in self.status.items():
                statusTxt += f"{k} {v}\n"
            if not self.auto and self.tg:
                print_at(statusTxt, tg=self.tg, print_allowed=True, reply_markup=self.tg.actionKeyboard(self.symbol, self.status['sl'], self.status['tp'], deviation=5, lot=self.LOT))
            elif self.auto:
                # define the action type
                if self.status['type'] == 'rise':
                    actionType = 'long'
                else:
                    actionType = 'short'
                # build request format
                request = self.mt5Controller.executor.request_format(
                    symbol=self.symbol,
                    actionType=actionType,
                    sl=float(self.status['sl']),
                    tp=float(self.status['tp']),
                    deviation=5,
                    lot=self.LOT
                )
                # execute request
                self.mt5Controller.executor.request_execute(request)
                self.breakThroughTime = masterSignal.index[-1]  # save the last toke action time
                self.inPosition = True
        # reset the notice if in next time slot
        if self.inPosition:
            lastPrice = masterSignal.iloc[-1]['close']
            if (self.breakThroughTime != masterSignal.index[-1]): # not at the same time
                if (self.trendType == 'rise' and (lastPrice >= self.status['tp'] or lastPrice <= self.status['sl'])) or (self.trendType == 'down' and (lastPrice <= self.status['tp'] or lastPrice >= self.status['sl'])):
                    self.inPosition = False

    def run(self):
        while True:
            time.sleep(5)
            # getting latest Prices
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[self.symbol],
                                                               start=None,
                                                               end=None,
                                                               timeframe='5min',
                                                               count=1000,
                                                               ohlcvs='111111'
                                                               )
            # getting ohlcvs
            ohlcvs = Prices.getOhlcvsFromPrices()[self.symbol]

            # getting master signal
            masterSignal = self.getMasterSignal(ohlcvs, self.lowerEma, self.middleEma, self.upperEma,
                                                self.diff_ema_upper_middle, self.diff_ema_middle_lower,
                                                self.ratio_sl_sp, needEarning=False)

            self.checkValidAction(masterSignal, self.trendType)

# get live Data from MT5 Server
# mt5Controller = Mt5Controller()
# swingScalping = SwingScalping(mt5Controller, 'USDJPY', breakThroughCondition='50')
# swingScalping.run()
