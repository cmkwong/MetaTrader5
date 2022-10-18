import telebot
from telebot.callback_data import CallbackData, CallbackDataFilter
from telebot import types

import inspect

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.StrategyController import StrategyController
from myUtils import paramModel


class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.bot = telebot.TeleBot(token)
        self.mt5Controller = MT5Controller()
        self.strategyController = StrategyController(self.mt5Controller, self.bot)

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

        @self.bot.message_handler(commands=['add'])
        def selectStrategy(message):
            self.chat_id = message.chat.id
            txt = self.strategyController.getListStrategiesText()
            msg = self.bot.reply_to(message, txt + "\nEnter Strategy Number: ")
            self.bot.register_next_step_handler(msg, choose_strategy)

        def choose_strategy(message):
            strategy_index = message.text
            if not strategy_index.isdigit():
                self.bot.send_message(message.chat.id, "This is not a number")
                return False
            strategy_index = int(strategy_index)
            # sig = inspect.signature(self.strategyController.listStrategies[strategy_index])
            # for param in sig.parameters.values():
            #     if (param.kind == param.KEYWORD_ONLY):
            #         print(param.name, param.default)
            msg = self.bot.reply_to(message, "Please Choose Symbol: ")
            self.bot.register_next_step_handler(msg, symbols_command_handler)

        @self.bot.message_handler(commands=['symbols'])
        def symbols_command_handler(message: types.Message):
            self.bot.send_message(message.chat.id, 'Symbols:', reply_markup=self.symbolKeyboard())

        self.bot.polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
