import telebot
from telebot.callback_data import CallbackData, CallbackDataFilter
from telebot.custom_filters import AdvancedCustomFilter
from telebot import types

import inspect

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.StrategyController import StrategyController
from myUtils import paramModel


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
        self.strategyController = StrategyController(self.mt5Controller, self.bot)
        self.strategy_factory = CallbackData('strategy_id', prefix='strategy')
        self.action_factory = CallbackData('action_id', 'symbol', 'sl', 'tp', 'deviation', 'lot', 'msg', prefix='action')
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

    def symbolKeyboard(self):
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

    def actionKeyboard(self, symbol, sl, tp, deviation, lot, msg):
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
                                                              lot=lot,
                                                              msg=msg)
                    )
                ]
                for action in self.ListAction
            ]
        )

    def run(self):
        # -------------------- Strategy --------------------
        @self.bot.message_handler(commands=['strategy'])
        def strategy_command_handler(message):
            self.chat_id = message.chat.id
            self.bot.send_message(message.chat.id, "Strategies: ", reply_markup=self.listStrategyKeyboard())

        @self.bot.callback_query_handler(func=None, config=self.strategy_factory.filter())
        def choose_strategy_callback(call):
            print('yeah')
            self.bot.answer_callback_query(callback_query_id=call.id, text='yeah', show_alert=True)

        @self.bot.message_handler(commands=['symbols'])
        def symbols_command_handler(message: types.Message):
            self.bot.send_message(message.chat.id, 'Symbols:', reply_markup=self.symbolKeyboard())

        # -------------------- Action --------------------
        @self.bot.callback_query_handler(func=None, config=self.action_factory.filter(action_id='2'))  # CANCEL
        def choose_strategy_callback(call):
            # getting callback data
            callback_data: dict = self.action_factory.parse(callback_data=call.data)
            self.bot.edit_message_text(chat_id=self.chat_id, message_id=call.message.message_id, text=callback_data['msg'] + '\nDeal Cancelled')

        @self.bot.callback_query_handler(func=None, config=self.action_factory.filter())  # LONG / SHORT
        def choose_strategy_callback(call):
            # getting callback data
            callback_data: dict = self.action_factory.parse(callback_data=call.data)
            requiredAction = None
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
            self.mt5Controller.executor.request_execute(request)
            self.bot.edit_message_text(chat_id=self.chat_id, message_id=call.message.message_id, text=callback_data['msg'] + '\nDone')

        # -------------------- Self defined run ---------------------
        @self.bot.message_handler(commands=['run'])
        def run_command_handler(message):
            self.chat_id = message.chat.id
            self.strategyController.runThreadStrategy(0, 'GBPUSD', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'CADJPY', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'AUDJPY', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'AUDUSD', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'USDCAD', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'USDJPY', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'EURCAD', breakThroughCondition='50', tg=self)
            # self.strategyController.runThreadStrategy(0, 'EURUSD', breakThroughCondition='50', tg=self)

        self.bot.add_custom_filter(StrategyCallbackFilter())
        self.bot.add_custom_filter(ActionCallbackFilter())
        self.bot.polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
