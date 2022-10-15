import sys

sys.path.append('C:/Users/Chris/projects/210215_mt5')
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Backtest import techModel
from myUtils.printModel import print_at

import pandas as pd
import time


class SwingScalping:
    def __init__(self, mt5Controller, symbol, diff_ema_100_50=45, diff_ema_50_25=30, ratio_sl_sp=1.5, tg=None):
        # define the controller
        self.mt5Controller = mt5Controller
        self.symbol = symbol
        self.diff_ema_100_50 = diff_ema_100_50
        self.diff_ema_50_25 = diff_ema_50_25
        self.ratio_sl_sp = ratio_sl_sp
        self.actionBreakThrough25_50 = 0  # 0 = 25; 1 = 50
        # init the variables
        self.breakUpThroughTime, self.breakDownThroughTime = None, None
        self.breakUpThroughCondition, self.breakUpThroughNotice, self.downTrendRangeCondition = False, False, False
        self.breakDownThroughCondition, self.breakDownThroughNotice, self.riseTrendRangeCondition = False, False, False
        # define tg
        self.tg = tg
        self.loop_tg = True

    @property
    def getName(self):
        return f"{self.__class__.__name__}_{self.symbol}"

    def run(self, update, context):

        while (self.loop_tg):
            time.sleep(5)
            # getting the live price
            Prices = self.mt5Controller.mt5PricesLoader.getPrices(symbols=[self.symbol],
                                                                  start=None,
                                                                  end=None,
                                                                  timeframe='5min',
                                                                  count=1000,
                                                                  ohlcvs='111111'
                                                                  )
            # get the close price
            close = Prices.c.values.reshape(-1)
            currentTime = Prices.c.index[-1]

            # calculate the ema price: 25, 50, 100
            ema = pd.DataFrame(index=Prices.c.index)
            ema['25'] = techModel.get_EMA(Prices.c, 25)
            ema['50'] = techModel.get_EMA(Prices.c, 50)
            ema['100'] = techModel.get_EMA(Prices.c, 100)

            # get latest 3 close price
            latest3Close = close[-3]
            # get latest 2 close price
            latest2Close = close[-2]
            # get latest close price
            latest1Close = close[-1]

            # calculate the ema difference
            digits = self.mt5Controller.all_symbol_info[self.symbol].digits
            # pt_values = self.mt5Controller.all_symbol_info[self.symbol].pt_value
            ptDiff_100_50 = (ema['100'] - ema['50']) * (10 ** digits)
            ptDiff_50_25 = (ema['50'] - ema['25']) * (10 ** digits)

            msg = ''
            msg += f"--------------------{currentTime}__{self.symbol}--------------------" + '\n'
            msg += "{:>15}{:>15}{:>15}".format('latest', 'latest 2', 'latest 3') + '\n'
            msg += "{:>15}{:>15}{:>15}".format(f"{latest1Close:.5f}", f"{latest2Close:.5f}", f"{latest3Close:.5f}") + '\n'
            msg += "{:>15}{:>15}{:>15}".format('EMA100', 'EMA50', 'EMA25') + '\n'
            msg += "{:>15}{:>15}{:>15}".format(f"{ema['100'][-1]:.5f}", f"{ema['50'][-1]:.5f}", f"{ema['25'][-1]:.5f}") + '\n'
            msg += f"EMA100-EMA50: {ptDiff_100_50[-1]:.2f}" + '\n'
            msg += f" EMA50-EMA25: {ptDiff_50_25[-1]:.2f}" + '\n'

            self.tg.preActingNotice(update, context, msg)

            # check range between EMA
            self.downTrendRangeCondition = (ptDiff_100_50[-1] >= self.diff_ema_100_50) and (ptDiff_50_25[-1] >= self.diff_ema_50_25)
            self.riseTrendRangeCondition = (ptDiff_100_50[-1] <= -self.diff_ema_100_50) and (ptDiff_50_25[-1] <= -self.diff_ema_50_25)

            # if range has larger specific value, enter into monitoring mode
            if self.downTrendRangeCondition:
                # --------------------------- DOWN TREND ---------------------------
                # check last-2-close price is below than 25-ema AND last-1-close price is larger than 25-ema
                if self.actionBreakThrough25_50 == 0:
                    self.breakUpThroughCondition = (latest3Close < ema['25'][-1]) and (latest2Close > ema['25'][-1])
                # check last-2-close price is below than 50-ema AND last-1-close price is larger than 50-ema
                else:
                    self.breakUpThroughCondition = (latest3Close < ema['50'][-1]) and (latest2Close > ema['50'][-1])

                # notice to user 50-ema being broken
                if self.breakUpThroughCondition and not self.breakUpThroughNotice:
                    # calculate the stop loss and stop profit
                    stopLoss = ema['100'][-1]
                    stopProfit = latest1Close - (ema['100'][-1] - latest1Close) * self.ratio_sl_sp
                    # make notice text to TG
                    # decide to action
                    # print out result to TG, if trade finished
                    # update all Data into database (for training purpose)
                    self.breakUpThroughTime = currentTime
                    print_at(f'Down Trend\n Time: {self.breakUpThroughTime}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n', self.tg)
                    self.breakUpThroughNotice = True
                    # print(f"{currentTime}, {self.breakUpThroughTime}, {currentTime is not self.breakUpThroughTime}")
            # reset the notice if in next time slot
            if self.breakUpThroughCondition and (currentTime != self.breakUpThroughTime):
                self.breakUpThroughNotice = False

            # if range has larger specific value, enter into monitoring mode
            if self.riseTrendRangeCondition:
                # --------------------------- RISE TREND ---------------------------
                # check last-2-close price is larger than 25-ema AND last-1-close price is below than 25-ema
                if self.actionBreakThrough25_50 == 0:
                    self.breakDownThroughCondition = (latest3Close > ema['25'][-1]) and (latest2Close < ema['25'][-1])
                # check last-2-close price is larger than 50-ema AND last-1-close price is below than 50-ema
                else:
                    self.breakDownThroughCondition = (latest3Close > ema['50'][-1]) and (latest2Close < ema['50'][-1])

                # notice to user 50-ema being broken
                if self.breakDownThroughCondition and not self.breakDownThroughNotice:
                    # calculate the stop loss and stop profit
                    stopLoss = ema['100'][-1]
                    stopProfit = latest1Close + (latest1Close - ema['100'][-1]) * self.ratio_sl_sp
                    # make notice text to TG
                    # decide to action
                    # print out result to TG, if trade finished
                    # update all Data into database (for training purpose)
                    self.breakDownThroughTime = currentTime
                    print_at(f'Rise Trend\n Time: {self.breakDownThroughTime}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n', self.tg)
                    self.breakDownThroughNotice = True
                    # print(f"{currentTime}, {self.breakDownThroughTime}, {currentTime is not self.breakDownThroughTime}")
            # reset the notice if in next time slot
            if self.breakDownThroughCondition and (currentTime != self.breakDownThroughTime):
                self.breakDownThroughNotice = False


# get live Data from MT5 Server
# mt5Controller = MT5Controller()
# swingScalping = SwingScalping(mt5Controller, 'USDJPY')
# swingScalping.run()
