from myUtils.printModel import print_at
from mt5Server.codes.Strategies.SwingScalping.Live_SwingScalping import Live_SwingScalping

from myDataFeed.myMt5.MT5Controller import MT5Controller
# from mt5Server.codes.Tg.TgController_pytb import Telegram_Bot
import threading

class StrategyController:
    def __init__(self, mt5Controller, tg=None):
        self.mt5Controller = mt5Controller
        self.runningStrategies = []
        self.idleStrategies = {}    # ['name', 'strategy id']
        self.listStrategies = [
            {'id': 0, 'name': Live_SwingScalping.__name__, 'class': Live_SwingScalping}
        ]
        self.Sybmols = ['GBPUSD','CADJPY','AUDJPY','AUDUSD','USDCAD','USDJPY','EURCAD','EURUSD']
        self.tg = tg

    # get list of strategies text
    def getListStrategiesText(self):
        txt = ''
        for id, strategy in enumerate(self.listStrategies):
            txt += f"{id}. {strategy['name']}\n"
        return txt

    # run strategy
    def runThreadStrategy(self, strategyId, symbol, **kwargs):
        for strategy in self.listStrategies:
            if strategy['id'] == strategyId:
                targetStrategy = strategy['class'](self.mt5Controller, symbol, **kwargs)
                thread = threading.Thread(target=targetStrategy.run)
                thread.start()
                self.runningStrategies.append(targetStrategy)


# mt5Controller = MT5Controller()
# telegram_Bot = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
# strategyController = StrategyController(mt5Controller)
# strategyController.runThreadStrategy(0, 'USDJPY')
# strategyController.runThreadStrategy(0, 'EURUSD', breakThroughCondition='50')
# strategyController.runThreadStrategy(0, 'AUDUSD', breakThroughCondition='50')
# strategyController.runThreadStrategy(0, 'AUDJPY', breakThroughCondition='50')
# strategyController.runThreadStrategy(0, 'GBPUSD', breakThroughCondition='50')

