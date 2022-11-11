import pandas as pd

from mt5Server.codes.Backtest.func import techModel

class Base_SwingScalping:
    def __init__(self, mt5Controller, symbol):
        # define the controller
        self.mt5Controller = mt5Controller
        self.symbol = symbol

    # calculate the ema difference
    def getPointDiff(self, symbol, upper, middle, lower):
        digits = self.mt5Controller.all_symbol_info[symbol].digits
        return (upper - middle) * (10 ** digits), (middle - lower) * (10 ** digits)

    # get break through signal
    def getBreakThroughSignal(self, ohlc: pd.DataFrame, ema: pd.DataFrame):
        # shiftedClose = pd.DataFrame()
        ema['latest1Close'] = ohlc['close']
        ema['latest2Close'] = ohlc['close'].shift(1)
        ema['latest3Close'] = ohlc['close'].shift(2)
        ema['riseBreak'] = (ema['latest2Close'] < ema['middle']) & (ema['latest3Close'] > ema['middle'])
        ema['downBreak'] = (ema['latest2Close'] > ema['middle']) & (ema['latest3Close'] < ema['middle'])
        return ema.loc[:, 'riseBreak'], ema.loc[:, 'downBreak']

    def getMasterSignal(self, ohlc, lowerEma, middleEma, upperEma, diff_ema_upper_middle, diff_ema_middle_lower, ratio_sl_sp):
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
        signal['open'] = ohlc.open
        signal['high'] = ohlc.high
        signal['low'] = ohlc.low
        signal['close'] = ohlc.close

        # calculate the ema bandwidth
        signal['lower'] = techModel.get_EMA(ohlc.close, lowerEma)
        signal['middle'] = techModel.get_EMA(ohlc.close, middleEma)
        signal['upper'] = techModel.get_EMA(ohlc.close, upperEma)

        # calculate the points difference
        signal['ptDiff_upper_middle'], signal['ptDiff_middle_lower'] = self.getPointDiff(self.symbol, signal['upper'], signal['middle'], signal['lower'])

        # get break through signal
        signal['riseBreak'], signal['downBreak'] = self.getBreakThroughSignal(ohlc.loc[:, ('open', 'high', 'low', 'close')], signal.loc[:, ('lower', 'middle', 'upper')])

        # get trend range conditions
        signal['riseRange'] = (signal['ptDiff_upper_middle'] <= -diff_ema_upper_middle) & (signal['ptDiff_middle_lower'] <= -diff_ema_middle_lower)
        signal['downRange'] = (signal['ptDiff_upper_middle'] >= diff_ema_upper_middle) & (signal['ptDiff_middle_lower'] >= diff_ema_middle_lower)

        # stop loss
        signal['stopLoss'] = signal['upper']

        # take profit
        signal['takeProfit'] = signal['close'] - (signal['upper'] - signal['close']) * ratio_sl_sp

        return signal
