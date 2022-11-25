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
                 # rise
                 rise_diff_ema_upper_middle=45, rise_diff_ema_middle_lower=30, rise_ratio_sl_sp=1.5,
                 rise_lowerEma=25, rise_middleEma=50, rise_upperEma=100,
                 # down
                 down_diff_ema_upper_middle=45, down_diff_ema_middle_lower=30, down_ratio_sl_sp=1.5,
                 down_lowerEma=25, down_middleEma=50, down_upperEma=100,
                 lot=1,
                 auto=False, tg=None):
        super(SwingScalping, self).__init__(mt5Controller, symbol)
        self.RISE_PARAM = {}
        self.RISE_PARAM['diff_ema_upper_middle'] = rise_diff_ema_upper_middle
        self.RISE_PARAM['diff_ema_middle_lower'] = rise_diff_ema_middle_lower
        self.RISE_PARAM['ratio_sl_sp'] = rise_ratio_sl_sp
        # self.RISE_PARAM['break_through_condition'] = rise_breakThroughCondition  # 0 = 25; 1 = 50
        self.RISE_PARAM['lowerEma'] = rise_lowerEma
        self.RISE_PARAM['middleEma'] = rise_middleEma
        self.RISE_PARAM['upperEma'] = rise_upperEma

        self.DOWN_PARAM = {}
        self.DOWN_PARAM['diff_ema_upper_middle'] = down_diff_ema_upper_middle
        self.DOWN_PARAM['diff_ema_middle_lower'] = down_diff_ema_middle_lower
        self.DOWN_PARAM['ratio_sl_sp'] = down_ratio_sl_sp
        # self.DOWN_PARAM['break_through_condition'] = down_breakThroughCondition  # 0 = 25; 1 = 50
        self.DOWN_PARAM['lowerEma'] = down_lowerEma
        self.DOWN_PARAM['middleEma'] = down_middleEma
        self.DOWN_PARAM['upperEma'] = down_upperEma
        # lot
        self.LOT = lot
        # init the variables
        self.breakThroughTime = None
        self.breakThroughCondition, self.trendRangeCondition = False, False
        # check if notice have been announced
        self.actionMade = {}
        self.actionMade['rise'] = False
        self.actionMade['down'] = False
        # auto
        self.auto = auto
        # define tg
        self.tg = tg

    @property
    def getName(self):
        return f"{self.__class__.__name__}_{self.symbol}: {self.RISE_PARAM['diff_ema_upper_middle']} {self.RISE_PARAM['diff_ema_middle_lower']} {self.RISE_PARAM['ratio_sl_sp']} {self.RISE_PARAM['break_through_condition']}"

    def gettingEmaDiff_DISCARD(self, mute=False):
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
        EmaDiff.ema['25'] = techModel.get_EMA(Prices.c, self.RISE_PARAM['lowerEma'])
        EmaDiff.ema['50'] = techModel.get_EMA(Prices.c, self.RISE_PARAM['middleEma'])
        EmaDiff.ema['100'] = techModel.get_EMA(Prices.c, self.RISE_PARAM['upperEma'])

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
    def checkBreakThrough_DISCARD(self, EmaDiff, trendType='down'):
        # check range condition
        if trendType == 'down':
            self.trendRangeCondition = (EmaDiff.ptDiff_100_50[-1] >= self.RISE_PARAM['diff_ema_upper_middle']) and (EmaDiff.ptDiff_50_25[-1] >= self.RISE_PARAM['diff_ema_middle_lower'])
            self.breakThroughCondition = (EmaDiff.latest3Close < EmaDiff.ema[self.RISE_PARAM['break_through_condition']][-1]) and (EmaDiff.latest2Close > EmaDiff.ema[self.RISE_PARAM['break_through_condition']][-1])
        elif trendType == 'rise':
            self.trendRangeCondition = (EmaDiff.ptDiff_100_50[-1] <= -self.RISE_PARAM['diff_ema_upper_middle']) and (EmaDiff.ptDiff_50_25[-1] <= -self.RISE_PARAM['diff_ema_middle_lower'])
            self.breakThroughCondition = (EmaDiff.latest3Close > EmaDiff.ema[self.RISE_PARAM['break_through_condition']][-1]) and (EmaDiff.latest2Close < EmaDiff.ema[self.RISE_PARAM['break_through_condition']][-1])

        # if range has larger specific value, enter into monitoring mode
        # if already noticed in same time-slot, return the function
        if self.trendRangeCondition and self.breakThroughCondition and not self.actionMade:
            self.breakThroughTime = EmaDiff.currentTime
            # calculate the stop loss and stop profit
            stopLoss = EmaDiff.ema['100'][-1]
            takeProfit = EmaDiff.latest1Close - (EmaDiff.ema['100'][-1] - EmaDiff.latest1Close) * self.RISE_PARAM['ratio_sl_sp']
            # print(f'Trend({trendType})\n Time: {self.breakThroughTime}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n')
            self.actionMade = True
            status = {'type': trendType, 'time': self.breakThroughTime, 'sl': stopLoss, 'tp': takeProfit}
            self.makeAction_DISCARD(status)
            return status

        # reset the notice if in next time slot
        if EmaDiff.currentTime != self.breakThroughTime:
            self.actionMade = False

        return False

    def makeAction_DISCARD(self, status):
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

    def run_DISCARD(self):
        while True:
            # status = {'type': 'rise', 'time': '', 'sl': 100, 'tp': 50} # testing
            # self.makeNotice(status)
            time.sleep(5)
            # getting the live price
            EmaDiff = self.gettingEmaDiff_DISCARD(mute=True)
            # --------------------------- DOWN TREND ---------------------------
            status = self.checkBreakThrough_DISCARD(EmaDiff, 'down')
            if status: return status
            # --------------------------- RISE TREND ---------------------------
            status = self.checkBreakThrough_DISCARD(EmaDiff, 'rise')
            if status: return status

    def checkValidAction(self, masterSignal, trendType='rise'):
        lastRow = masterSignal.iloc[-1, :]
        if (lastRow[trendType + 'Break'] and lastRow[trendType + 'Range'] and not self.actionMade[trendType]):
            status = {'type': trendType, 'time': self.breakThroughTime, 'sl': lastRow['stopLoss'], 'tp': lastRow['takeProfit']}

            # make notice and take action if auto set to True
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
            riseMasterSignal = self.getMasterSignal(ohlcvs, self.RISE_PARAM['lowerEma'], self.RISE_PARAM['middleEma'], self.RISE_PARAM['upperEma'],
                                                    self.RISE_PARAM['diff_ema_upper_middle'], self.RISE_PARAM['diff_ema_middle_lower'],
                                                    self.RISE_PARAM['ratio_sl_sp'], needEarning=False)

            downMasterSignal = self.getMasterSignal(ohlcvs, self.DOWN_PARAM['lowerEma'], self.DOWN_PARAM['middleEma'], self.DOWN_PARAM['upperEma'],
                                                    self.DOWN_PARAM['diff_ema_upper_middle'], self.DOWN_PARAM['diff_ema_middle_lower'],
                                                    self.DOWN_PARAM['ratio_sl_sp'], needEarning=False)

            self.checkValidAction(riseMasterSignal, 'rise')
            self.checkValidAction(downMasterSignal, 'down')

# get live Data from MT5 Server
# mt5Controller = MT5Controller()
# swingScalping = SwingScalping(mt5Controller, 'USDJPY', breakThroughCondition='50')
# swingScalping.run()
