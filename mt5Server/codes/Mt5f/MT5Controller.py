from mt5Server.codes.Mt5f.loader.MT5PricesLoader import MT5PricesLoader
from mt5Server.codes.Mt5f.executor.MT5Executor import MT5Executor
from mt5Server.codes.Mt5f.BaseMt5 import BaseMt5


class MT5Controller(BaseMt5):
    def __init__(self, timezone='Hongkong', deposit_currency='USD', type_filling='ioc'):
        super().__init__()
        self.executor = MT5Executor(type_filling)  # execute the request (buy/sell)
        self.pricesLoader = MT5PricesLoader(self.all_symbol_info, timezone, deposit_currency)  # loading the loader
