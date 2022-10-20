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

class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.bot = telebot.TeleBot(token)
        self.mt5Controller = MT5Controller()
        self.strategyController = StrategyController(self.mt5Controller, self.bot)
        self.strategy_factory = CallbackData('strategy_id', prefix='strategy')

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

    def run(self):

        @self.bot.message_handler(commands=['strategy'])
        def strategy_command_handler(message):
            self.chat_id = message.chat.id
            # txt = self.strategyController.getListStrategiesText()
            # msg = self.bot.reply_to(message, txt + "\nEnter Strategy Number: ")
            # self.bot.register_next_step_handler(msg, choose_strategy)
            self.bot.send_message(message.chat.id, "Strategies: ", reply_markup=self.listStrategyKeyboard())

        @self.bot.callback_query_handler(func=None, config=self.strategy_factory.filter())
        def choose_strategy_callback(call):
            print('yeah')
            self.bot.answer_callback_query(callback_query_id=call.id, text='yeah', show_alert=True)
            # strategy_index = message.text
            # if not strategy_index.isdigit():
            #     self.bot.send_message(message.chat.id, "This is not a number")
            #     return False
            # strategy_index = int(strategy_index)
            # sig = inspect.signature(self.strategyController.listStrategies[strategy_index])
            # for param in sig.parameters.values():
            #     if (param.kind == param.KEYWORD_ONLY):
            #         print(param.name, param.default)
            # msg = self.bot.reply_to(message, "Please Choose Symbol: ")
            # self.bot.register_next_step_handler(msg, symbols_command_handler)

        @self.bot.message_handler(commands=['symbols'])
        def symbols_command_handler(message: types.Message):
            self.bot.send_message(message.chat.id, 'Symbols:', reply_markup=self.symbolKeyboard())

        self.bot.add_custom_filter(StrategyCallbackFilter())
        self.bot.polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
