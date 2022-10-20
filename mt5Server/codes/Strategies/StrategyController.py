from myUtils.printModel import print_at
from mt5Server.codes.Strategies.Scalping.SwingScalping import SwingScalping

class StrategyController:
    def __init__(self, mt5Controller, tg=None):
        self.mt5Controller = mt5Controller
        self.runningStrategies = {}
        self.idleStrategies = {}
        self.listStrategies = [
            {'id': 0, 'name': SwingScalping.__name__}
        ]
        self.Sybmols = ['USDJPY', 'AUDUSD']
        self.tg = tg

    # get list of strategies text
    def getListStrategiesText(self):
        txt = ''
        for id, strategy in enumerate(self.listStrategies):
            txt += f"{id}. {strategy.__name__}\n"
        return txt

    # def _setStrategy(self, strategyId, symbol):
    #     strategy = self.distStrategies[strategyId](self.mt5Controller, symbol, tg=self.tg)
    #     self.idleStrategies[strategy.getName] = strategy
    #
    # def _runStrategy(self):
    #     txt = ''
    #     for id, strategy in self.distStrategies:
    #         txt += f"{id}. {strategy.__name__}\n"
    #     print_at(txt, tg=self.tg)

