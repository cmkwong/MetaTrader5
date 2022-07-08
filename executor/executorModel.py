from backtest import timeModel, pointsModel, returnModel
from executor import common as mt5common
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import collections
import os

"""
This is pure executor that will not store any status of trades
The status will be stored in NodeJS server, including:

self-defined:
    StrategyId

MetaTrader:
    positionId
    
"""

class Executor(mt5common.BaseMt5):
    def __init__(self, type_filling='ioc'):
        super().__init__(type_filling)


