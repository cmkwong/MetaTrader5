import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
sys.path.append('C:/Users/Chris/projects/AtomLib')

import telebot
from telebot.callback_data import CallbackData, CallbackDataFilter
from telebot.custom_filters import AdvancedCustomFilter
from telebot import types

from myDataFeed.myMt5.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.StrategyController import StrategyController
from myDataFeed.myNodeJs.DataFeeder import DataFeeder


class StrategyCallbackFilter(AdvancedCustomFilter):
    key = 'config'

    def check(self, call: types.CallbackQuery, config: CallbackDataFilter):
        return config.check(query=call)


class ActionCallbackFilter(AdvancedCustomFilter):
    key = 'config'

    def check(self, call: types.CallbackQuery, config: CallbackDataFilter):
        return config.check(query=call)


class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.bot = telebot.TeleBot(token)
        self.mt5Controller = MT5Controller()
        self.dataFeeder = DataFeeder(self.mt5Controller)
        self.strategyController = StrategyController(self.mt5Controller, self.bot)
        self.strategy_factory = CallbackData('strategy_id', prefix='strategy')
        self.action_factory = CallbackData('action_id', 'symbol', 'sl', 'tp', 'deviation', 'lot', prefix='action')
        self.ListAction = [
            {'id': '0', 'actionType': 'long'},
            {'id': '1', 'actionType': 'short'},
            {'id': '2', 'actionType': 'cancel'}
        ]

    def idleStrategyKeyboard(self):
        return types.InlineKeyboardMarkup(
            keyboard=[
                [
                    types.InlineKeyboardButton(
                        text=strategy['name'],
                        callback_data=self.strategy_factory.new(strategy_id=strategy['id'])
                    )
                ]
                for strategy in self.strategyController.idleStrategies
            ]
        )

    def listStrategyKeyboard(self):
        return types.InlineKeyboardMarkup(
            keyboard=[
                [
                    types.InlineKeyboardButton(
                        text=strategy['name'],
                        callback_data=self.strategy_factory.new(strategy_id=strategy['id'])
                    )
                ]
                for strategy in self.strategyController.listStrategies
            ]
        )

    def listSymbolKeyboard(self):
        return types.InlineKeyboardMarkup(
            keyboard=[
                [
                    types.InlineKeyboardButton(
                        text=symbol,
                        callback_data=symbol
                    )
                ]
                for symbol in self.strategyController.Sybmols
            ]
        )

    def actionKeyboard(self, symbol, sl, tp, deviation, lot):
        return types.InlineKeyboardMarkup(
            keyboard=[
                [
                    types.InlineKeyboardButton(
                        text=action['actionType'],
                        callback_data=self.action_factory.new(action_id=action['id'],
                                                              symbol=symbol,
                                                              sl=sl,
                                                              tp=tp,
                                                              deviation=deviation,
                                                              lot=lot
                                                              )
                    )
                ]
                for action in self.ListAction
            ]
        )

    def run(self):
        # -------------------- Strategy --------------------
        @self.bot.message_handler(commands=['strategy'])
        def strategy_command_handler(message):
            self.bot.send_message(message.chat.id, "Strategies: ", reply_markup=self.listStrategyKeyboard())

        @self.bot.callback_query_handler(func=None, config=self.strategy_factory.filter())
        def choose_strategy_callback(call):
            self.bot.answer_callback_query(callback_query_id=call.id, text='yeah', show_alert=True)

        @self.bot.message_handler(commands=['symbols'])
        def symbols_command_handler(message: types.Message):
            self.bot.send_message(message.chat.id, 'Symbols:', reply_markup=self.listSymbolKeyboard())

        # -------------------- Running Strategy List --------------------
        @self.bot.message_handler(commands=['rl'])  # running strategy list
        def showRunStrategy_command_handler(message):
            txt = ''
            for i, strategy in enumerate(self.strategyController.runningStrategies):
                txt += f"{i + 1}. {strategy.getName}\n"
            self.bot.send_message(message.chat.id, f"Running Strategy: \n{txt}")

        # -------------------- Showing last deal result --------------------
        @self.bot.message_handler(commands=['dl'])  # historical deals list
        def showHistoricalDeal_command_handler(message):
            self.mt5Controller.get_historical_deal()

        @self.bot.message_handler(commands=['ol'])  # historical orders list
        def showHistoricalDeal_command_handler(message):
            self.mt5Controller.get_historical_order()

        # -------------------- Upload forex data into database (nodejs) --------------------
        @self.bot.message_handler(commands=['feed'])
        def feedDataIntoForex_command_handler(message):
            self.dataFeeder.uploadDatas(['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY'],
                                        startTime=(2022, 8, 31, 0, 0), endTime=(2022, 10, 31, 23, 59))

        # -------------------- Action Listener --------------------
        # Cancel
        @self.bot.callback_query_handler(func=None, config=self.action_factory.filter(action_id='2'))  # CANCEL
        def choose_strategy_callback(call):
            # getting callback data
            callback_data: dict = self.action_factory.parse(callback_data=call.data)
            # build msg
            msg = ''
            for k, v in callback_data.items():
                msg += f"{k} {v}\n"
            self.bot.edit_message_text(chat_id=self.chat_id, message_id=call.message.message_id, text=msg + '\nDeal Cancelled')

        # LONG / SHORT
        @self.bot.callback_query_handler(func=None, config=self.action_factory.filter())
        def choose_strategy_callback(call):
            # getting callback data
            callback_data: dict = self.action_factory.parse(callback_data=call.data)
            requiredAction = None

            # build request format
            for action in self.ListAction:
                if action['id'] == callback_data['action_id']:
                    requiredAction = action
                    break
            request = self.mt5Controller.executor.request_format(
                callback_data['symbol'],
                requiredAction['actionType'],
                float(callback_data['sl']),
                float(callback_data['tp']),
                int(callback_data['deviation']),
                int(callback_data['lot'])
            )
            # execute request
            self.mt5Controller.executor.request_execute(request)
            # build msg
            msg = ''
            for k, v in callback_data.items():
                msg += f"{k} {v}\n"
            self.bot.edit_message_text(chat_id=self.chat_id, message_id=call.message.message_id, text=msg + '\nDone')

        # -------------------- Self defined run ---------------------
        @self.bot.message_handler(commands=['run'])
        def run_command_handler(message):
            self.chat_id = message.chat.id
            self.strategyController.runThreadStrategy(0, 'GBPUSD',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=60, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=55, upperEma=92,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'GBPUSD',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=60, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=55, upperEma=92,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'USDJPY',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=60, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=55, upperEma=92,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'USDJPY',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=60, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=55, upperEma=92,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'AUDJPY',
                                                      diff_ema_upper_middle=40, diff_ema_middle_lower=20, ratio_sl_sp=2,
                                                      lowerEma=18, middleEma=39, upperEma=96,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'AUDJPY',
                                                      diff_ema_upper_middle=30, diff_ema_middle_lower=40, ratio_sl_sp=2.2,
                                                      lowerEma=26, middleEma=47, upperEma=88,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'CADJPY',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=20, ratio_sl_sp=2,
                                                      lowerEma=18, middleEma=31, upperEma=84,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'CADJPY',
                                                      diff_ema_upper_middle=20, diff_ema_middle_lower=20, ratio_sl_sp=1.8,
                                                      lowerEma=18, middleEma=47, upperEma=92,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'AUDUSD',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=20, ratio_sl_sp=2,
                                                      lowerEma=18, middleEma=31, upperEma=84,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'AUDUSD',
                                                      diff_ema_upper_middle=50, diff_ema_middle_lower=20, ratio_sl_sp=1.8,
                                                      lowerEma=18, middleEma=27, upperEma=56,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'USDCAD',
                                                      diff_ema_upper_middle=50, diff_ema_middle_lower=30, ratio_sl_sp=1.2,
                                                      lowerEma=18, middleEma=27, upperEma=48,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'USDCAD',
                                                      diff_ema_upper_middle=60, diff_ema_middle_lower=20, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=27, upperEma=68,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'EURUSD',
                                                      diff_ema_upper_middle=40, diff_ema_middle_lower=30, ratio_sl_sp=1.4,
                                                      lowerEma=18, middleEma=35, upperEma=80,
                                                      trendType='rise', lot=2,
                                                      auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'EURUSD',
                                                      diff_ema_upper_middle=60, diff_ema_middle_lower=20, ratio_sl_sp=2.2,
                                                      lowerEma=18, middleEma=31, upperEma=88,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)
            # self.strategyController.runThreadStrategy(0, 'EURCAD',
            #                                           diff_ema_upper_middle=40, diff_ema_middle_lower=30, ratio_sl_sp=1.4,
            #                                           lowerEma=18, middleEma=35, upperEma=80,
            #                                           trendType='rise', lot=2,
            #                                           auto=True, tg=self)
            self.strategyController.runThreadStrategy(0, 'EURCAD',
                                                      diff_ema_upper_middle=70, diff_ema_middle_lower=20, ratio_sl_sp=4.6,
                                                      lowerEma=26, middleEma=51, upperEma=92,
                                                      trendType='down', lot=2,
                                                      auto=True, tg=self)

            self.bot.send_message(message.chat.id, 'Strategy Running...')

        self.bot.add_custom_filter(StrategyCallbackFilter())
        self.bot.add_custom_filter(ActionCallbackFilter())
        self.bot.polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
