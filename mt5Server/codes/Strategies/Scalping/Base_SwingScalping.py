import pandas as pd
import numpy as np
import swifter

from mt5Server.codes.Backtest.func import techModel
from mt5Server.codes.Data.DataFeeder import DataFeeder


class Base_SwingScalping:
    def __init__(self, mt5Controller, symbol):
        # define the controller
        self.mt5Controller = mt5Controller
        self.symbol = symbol
        self.dataFeeder = DataFeeder(mt5Controller)
        self.digits = self.mt5Controller.all_symbol_info[symbol].digits
        self.pt_value = self.mt5Controller.all_symbol_info[symbol].pt_value

    # prepare for 1-minute data for further analysis
    def prepare1MinData(self, startTime, endTime):
        self.fetchData_min = self.dataFeeder.downloadData(self.symbol, startTime, endTime, timeframe='1min')

    # calculate the ema difference
    def getRangePointDiff(self, upper, middle, lower):
        return (upper - middle) * (10 ** self.digits), (middle - lower) * (10 ** self.digits)

    # get break through signal
    def getBreakThroughSignal(self, ohlc: pd.DataFrame, ema: pd.DataFrame):
        ema['latest1Close'] = ohlc['close']
        ema['latest2Close'] = ohlc['close'].shift(1)
        ema['latest3Close'] = ohlc['close'].shift(2)
        ema['riseBreak'] = (ema['latest2Close'] < ema['middle']) & (ema['latest3Close'] > ema['middle'])
        ema['downBreak'] = (ema['latest2Close'] > ema['middle']) & (ema['latest3Close'] < ema['middle'])
        return ema.loc[:, 'riseBreak'], ema.loc[:, 'downBreak']

    def getMasterSignal(self, fetchData_cust, lowerEma, middleEma, upperEma, diff_ema_upper_middle, diff_ema_middle_lower, ratio_sl_sp):
        """
        :param ohlc: pd.DataFrame
        :param lowerEma: int
        :param middleEma: int
        :param upperEma: int
        :param diff_ema_upper_middle: int
        :param diff_ema_middle_lower: int
        :param ratio_sl_sp: float
        :return: pd.DataFrame
        """
        signal = pd.DataFrame()
        signal['open'] = fetchData_cust.open
        signal['high'] = fetchData_cust.high
        signal['low'] = fetchData_cust.low
        signal['close'] = fetchData_cust.close
        signal['quote_exchg'] = fetchData_cust.quote_exchg


        # calculate the ema bandwidth
        signal['lower'] = techModel.get_EMA(fetchData_cust.close, lowerEma)
        signal['middle'] = techModel.get_EMA(fetchData_cust.close, middleEma)
        signal['upper'] = techModel.get_EMA(fetchData_cust.close, upperEma)

        # calculate the points difference
        signal['ptDiff_upper_middle'], signal['ptDiff_middle_lower'] = self.getRangePointDiff(signal['upper'], signal['middle'], signal['lower'])

        # get break through signal
        signal['riseBreak'], signal['downBreak'] = self.getBreakThroughSignal(fetchData_cust.loc[:, ('open', 'high', 'low', 'close')], signal.loc[:, ('lower', 'middle', 'upper')])

        # get trend range conditions
        signal['riseRange'] = (signal['ptDiff_upper_middle'] <= -diff_ema_upper_middle) & (signal['ptDiff_middle_lower'] <= -diff_ema_middle_lower)
        signal['downRange'] = (signal['ptDiff_upper_middle'] >= diff_ema_upper_middle) & (signal['ptDiff_middle_lower'] >= diff_ema_middle_lower)

        # stop loss
        signal['stopLoss'] = signal['upper']

        # take profit
        signal['takeProfit'] = signal['open'] - (signal['upper'] - signal['open']) * ratio_sl_sp

        return signal

    def getEarning(self, currentTime, breakCondition, rangeCondition, actionPrice, quote_exchg: float, sl: float, tp: float, trendType='rise'):
        # if str(currentTime) == '2022-09-22 17:35:00' and trendType == 'down':
        #     print('debug')
        if not breakCondition or not rangeCondition:
            return 0.0
        # rise trend
        if trendType == 'rise':
            last_tp = (self.fetchData_min.high >= tp)
            last_sl = (self.fetchData_min.low <= sl)
        # down trend
        else:
            last_tp = self.fetchData_min.low <= tp
            last_sl = self.fetchData_min.high >= sl
        tpTime = last_tp[currentTime:].eq(True).idxmax()
        slTime = last_sl[currentTime:].eq(True).idxmax()
        if tpTime < slTime and tpTime != currentTime:
            return np.abs(tp - actionPrice) * (10 ** self.digits) * quote_exchg * self.pt_value
        else:
            return -np.abs(sl - actionPrice) * (10 ** self.digits) * quote_exchg * self.pt_value
