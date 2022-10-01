import sys

sys.path.append('C:/Users/Chris/projects/210215_mt5')
from mt5Server.codes import config
from mt5Server.codes.Strategies.AI import common
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.backtest import techModel
from mt5Server.codes.backtest import pointsModel

import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time

def SwingScalping(symbol, diff_ema_100_50=60, diff_ema_50_25=30, ratio_sl_sp=1.5):

    # define the controller
    mt5Controller = MT5Controller()

    # init the variables
    ema25BreakUpThroughCondition, ema50BreakUpThroughCondition, ema25BreakUpThroughNotice, ema50BreakUpThroughNotice, downTrendRangeCondition = False, False, False, False, False
    ema25BreakDownThroughCondition, ema50BreakDownThroughCondition, ema25BreakDownThroughNotice, ema50BreakDownThroughNotice, riseTrendRangeCondition = False, False, False, False, False

    while(True):
        time.sleep(5)
        # getting the live price
        Prices = mt5Controller.mt5PricesLoader.getPrices(symbols=[symbol],
                                                         start=None,
                                                         end=None,
                                                         timeframe='5min',
                                                         count=200,
                                                         ohlcvs='111111'
                                                         )
        # get the close price
        close = Prices.c.values.reshape(-1)
        print(close.shape)
        # calculate the ema price: 25, 50, 100
        ema = pd.DataFrame(index=Prices.c.index)
        ema['25'] = techModel.get_EMA(close, 25)
        ema['50'] = techModel.get_EMA(close, 50)
        ema['100'] = techModel.get_EMA(close, 100)

        # get latest 3 close price
        latest3Close = close[-3]
        # get latest 2 close price
        latest2Close = close[-2]
        # get latest close price
        latest1Close = close[-1]

        # calculate the ema difference
        digits = mt5Controller.all_symbol_info[symbol].digits
        pt_values = mt5Controller.all_symbol_info[symbol].pt_value
        ptDiff_100_50 = (ema['100'] - ema['50']) * (10 ** digits)
        ptDiff_50_25 = (ema['50'] - ema['25']) * (10 ** digits)

        # check range between EMA
        downTrendRangeCondition = (ptDiff_100_50[-1] >= diff_ema_100_50) and (ptDiff_50_25[-1] >= diff_ema_50_25)
        riseTrendRangeCondition = (ptDiff_100_50[-1] <= -diff_ema_100_50) and (ptDiff_50_25[-1] <= -diff_ema_50_25)

        # if range has larger specific value, enter into monitoring mode
        if downTrendRangeCondition:
            # --------------------------- DOWN TREND ---------------------------
            # check last-2-close price is below than 25-ema AND last-1-close price is larger than 25-ema
            ema25BreakUpThroughCondition = (latest3Close < ema['25'][-1]) and (latest2Close > ema['25'][-1])
            # check last-2-close price is below than 50-ema AND last-1-close price is larger than 50-ema
            ema50BreakUpThroughCondition = (latest3Close < ema['50'][-1]) and (latest2Close > ema['50'][-1])

            # notice to user 50-ema being broken
            if ema50BreakUpThroughCondition and not ema50BreakUpThroughNotice:
                # calculate the stop loss and stop profit
                stopLoss = ema['100'][-1]
                stopProfit = latest1Close - (ema['100'][-1] - latest1Close) * ratio_sl_sp
                # make notice text to TG
                # decide to action
                # print out result to TG, if trade finished
                # update all Data into database (for training purpose)
                print(f'Down Trend\n Time: {close[-1].index}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n')
                ema50BreakUpThroughNotice = True
        # condition not existed, reset the status
        else:
            ema25BreakUpThroughNotice, ema50BreakUpThroughNotice = False, False

        # if range has larger specific value, enter into monitoring mode
        if riseTrendRangeCondition:
            # --------------------------- RISE TREND ---------------------------
            # check last-2-close price is larger than 25-ema AND last-1-close price is below than 25-ema
            ema25BreakDownThroughCondition = (latest3Close > ema['25'][-1]) and (latest2Close < ema['25'][-1])
            # check last-2-close price is larger than 50-ema AND last-1-close price is below than 50-ema
            ema50BreakDownThroughCondition = (latest3Close > ema['50'][-1]) and (latest2Close < ema['50'][-1])

            # notice to user 50-ema being broken
            if ema50BreakDownThroughCondition and not ema50BreakDownThroughNotice:
                # calculate the stop loss and stop profit
                stopLoss = ema['100'][-1]
                stopProfit = latest1Close + (latest1Close - ema['100'][-1]) * ratio_sl_sp
                # make notice text to TG
                # decide to action
                # print out result to TG, if trade finished
                # update all Data into database (for training purpose)
                print(f'Rise Trend\n Time: {close[-1].index}\n Stop loss: {stopLoss}\n Stop Profit: {stopProfit}\n\n')
                ema50BreakDownThroughNotice = True
        # condition not existed, reset the status
        else:
            ema25BreakDownThroughNotice, ema50BreakDownThroughNotice = False, False

# get live Data from MT5 Server
SwingScalping('AUDUSD')