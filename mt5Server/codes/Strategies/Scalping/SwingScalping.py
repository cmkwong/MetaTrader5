import sys
import os

sys.path.append('C:/Users/Chris/projects/210215_mt5')
sys.path.append('C:/Users/Chris/projects/AtomLib')
from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Strategies.Scalping.Base_SwingScalping import Base_SwingScalping
from myUtils.printModel import print_at

import pandas as pd
import collections
import time


class SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, symbol, *,
                 diff_ema_100_50=45, diff_ema_50_25=30, ratio_sl_sp=1.5, breakThroughCondition='25',
                 ema25Percent=25, ema50Percent=50, ema100Percent=100,
                 lot=1,
                 auto=False, tg=None):
        super(SwingScalping, self).__init__(mt5Controller, symbol)
        self.DIFF_EMA_100_50 = diff_ema_100_50
        self.DIFF_EMA_50_25 = diff_ema_50_25
        self.RATIO_SL_SP = ratio_sl_sp
        self.BREAK_THROUGH_CONDITION = breakThroughCondition  # 0 = 25; 1 = 50
        self.ema25Percent = ema25Percent
        self.ema50Percent = ema50Percent
        self.ema100Percent = ema100Percent
        self.LOT = lot
        # init the variables
        self.breakThroughTime = None
        self.breakThroughCondition, self.trendRangeCondition, self.breakThroughNoticed = False, False, False
        # auto
        self.auto = auto
        # define tg
        self.tg = tg

    @property
    def getName(self):
        return f"{self.__class__.__name__}_{self.symbol}: {self.DIFF_EMA_100_50} {self.DIFF_EMA_50_25} {self.RATIO_SL_SP} {self.BREAK_THROUGH_CONDITION}"

    def gettingEmaDiff(self, mute=False):
        # define the name tuple
        EmaDiff = collections.namedtuple('EmaDiff', ['currentTime', 'ema', 'ptDiff_100_50', 'ptDiff_50_25', 'latest3Close', 'latest2Close', 'latest1Close'])

        # getting latest Prices
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[self.symbol],
                                                           start=None,
                                                           end=None,
                                                           timeframe='5min',
                                                           count=1000,
                                                           ohlcvs='111111'
                                                           )
        # get the close price
        close = Prices.c.values.reshape(-1)
        EmaDiff.currentTime = Prices.c.index[-1]

        # calculate the ema price: 25, 50, 100
        EmaDiff.ema = pd.DataFrame(index=Prices.c.index)
        EmaDiff.ema['25'] = techModel.get_EMA(Prices.c, self.ema25Percent)
        EmaDiff.ema['50'] = techModel.get_EMA(Prices.c, self.ema50Percent)
        EmaDiff.ema['100'] = techModel.get_EMA(Prices.c, self.ema100Percent)

        # get latest 3 close price
        EmaDiff.latest3Close = close[-3]
        # get latest 2 close price
        EmaDiff.latest2Close = close[-2]
        # get latest close price
        EmaDiff.latest1Close = close[-1]

        # calculate the point difference
        EmaDiff.ptDiff_100_50, EmaDiff.ptDiff_50_25 = self.getRangePointDiff(EmaDiff.ema['100'], EmaDiff.ema['50'], EmaDiff.ema['25'])

        if not mute:
            msg = ''
            msg += f"--------------------{EmaDiff.currentTime}__{self.symbol}--------------------" + '\n'
            msg += "{:>15}{:>15}{:>15}".format('latest', 'latest 2', 'latest 3') + '\n'
            msg += "{:>15}{:>15}{:>15}".format(f"{EmaDiff.latest1Close:.5f}", f"{EmaDiff.latest2Close:.5f}", f"{EmaDiff.latest3Close:.5f}") + '\n'
            msg += "{:>15}{:>15}{:>15}".format('EMA100', 'EMA50', 'EMA25') + '\n'
            msg += "{:>15}{:>15}{:>15}".format(f"{EmaDiff.ema['100'][-1]:.5f}", f"{EmaDiff.ema['50'][-1]:.5f}", f"{EmaDiff.ema['25'][-1]:.5f}") + '\n'
            msg += f"EMA100-EMA50: {EmaDiff.ptDiff_100_50[-1]:.2f}" + '\n'
            msg += f" EMA50-EMA25: {EmaDiff.ptDiff_50_25[-1]:.2f}" + '\n'
            print(msg)
        return EmaDiff

    # check if condition is meet: down / rise trend
    def checkBreakThrough(self, EmaDiff, trendType='down'):
        # check range condition
        if trendType == 'down':
            self.trendRangeCondition = (EmaDiff.ptDiff_100_50[-1] >= self.DIFF_EMA_100_50) and (EmaDiff.ptDiff_50_25[-1] >= self.DIFF_EMA_50_25)
            self.breakThroughCondition = (EmaDiff.latest3Close < EmaDiff.ema[self.BREAK_THROUGH_CONDITION][-1]) and (EmaDiff.latest2Close > EmaDiff.ema[self.BREAK_THROUGH_CONDITION][-1])
        elif trendType == 'rise':
            self.trendRangeCondition = (EmaDiff.ptDiff_100_50[-1] <= -self.DIFF_EMA_100_50) and (EmaDiff.ptDiff_50_25[-1] <= -self.DIFF_EMA_50_25)
            self.breakThroughCondition = (EmaDiff.latest3Close > EmaDiff.ema[self.BREAK_THROUGH_CONDITION][-1]) and (EmaDiff.latest2Close < EmaDiff.ema[self.BREAK_THROUGH_CONDITION][-1])

        # if range has larger specific value, enter into monitoring mode
        # if already noticed in same time-slot, return the function
        if self.trendRangeCondition and self.breakThroughCondition and not self.breakThroughNoticed:
            self.breakThroughTime = EmaDiff.currentTime
            # calculate the stop loss and stop profit
            stopLoss = EmaDiff.ema['100'][-1]
            takeProfit = EmaDiff.latest1Close - (EmaDiff.ema['100'][-1] - EmaDiff.latest1Close) * self.RATIO_SL_SP
            # print(f'Trend({trendType})\n Time: {self.breakThroughTime}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n')
            self.breakThroughNoticed = True
            status = {'type': trendType, 'time': self.breakThroughTime, 'sl': stopLoss, 'tp': takeProfit}
            self.makeAction(status)
            return status

        # reset the notice if in next time slot
        if EmaDiff.currentTime != self.breakThroughTime:
            self.breakThroughNoticed = False

        return False

    def makeAction(self, status):
        os.system(f"start C:/Users/Chris/projects/210215_mt5/mt5Server/Sounds/{self.symbol}.mp3")
        statusTxt = f'{self.symbol}\n'
        for k, v in status.items():
            statusTxt += f"{k} {v}\n"
        if not self.auto and self.tg:
            print_at(statusTxt, tg=self.tg, print_allowed=True, reply_markup=self.tg.actionKeyboard(self.symbol, status['sl'], status['tp'], deviation=5, lot=self.LOT))
        elif self.auto:
            # define the action type
            if status['type'] == 'rise':
                actionType = 'long'
            else:
                actionType = 'short'
            # build request format
            request = self.mt5Controller.executor.request_format(
                symbol=self.symbol,
                actionType=actionType,
                sl=float(status['sl']),
                tp=float(status['tp']),
                deviation=5,
                lot=self.LOT
            )
            # execute request
            self.mt5Controller.executor.request_execute(request)

    def run(self):
        while True:
            # status = {'type': 'rise', 'time': '', 'sl': 100, 'tp': 50} # testing
            # self.makeNotice(status)
            time.sleep(5)
            # getting the live price
            EmaDiff = self.gettingEmaDiff(mute=True)
            # --------------------------- DOWN TREND ---------------------------
            status = self.checkBreakThrough(EmaDiff, 'down')
            if status: return status
            # --------------------------- RISE TREND ---------------------------
            status = self.checkBreakThrough(EmaDiff, 'rise')
            if status: return status

# get live Data from MT5 Server
# mt5Controller = MT5Controller()
# swingScalping = SwingScalping(mt5Controller, 'USDJPY', breakThroughCondition='50')
# swingScalping.run()
