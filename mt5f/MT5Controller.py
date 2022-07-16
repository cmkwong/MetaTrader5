from backtest import timeModel

from mt5f.loader.MT5PricesLoader import MT5PricesLoader
from mt5f.executor.MT5Executor import MT5Executor
from mt5f.BaseMt5 import BaseMt5


class MT5Controller(BaseMt5):
    def __init__(self, data_path='C:/Users/Chris/projects/210215_mt5/docs', timezone='Hongkong', deposit_currency='USD', type_filling='ioc'):
        super().__init__()
        self.mt5Executor = MT5Executor(type_filling)  # execute the request (buy/sell)
        self.mt5PricesLoader = MT5PricesLoader(self.all_symbol_info, data_path, timezone, deposit_currency)  # loading the loader
