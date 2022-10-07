# import logging
#
# from telegram import __version__ as TG_VER
#
# try:
#     from telegram import __version_info__
# except ImportError:
#     __version_info__ = (0, 0, 0, 0, 0)
#
# if __version_info__ < (20, 0, 0, "alpha", 1):
#     raise RuntimeError(
#         f"This example is not compatible with your current PTB version {TG_VER}. To view the "
#         f"{TG_VER} version of this example, "
#         f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
#     )

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
import threading
import asyncio

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.Scalping.SwingScalping import SwingScalping
from mt5Server.codes import config


class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.application = Application.builder().token(token).build()  # different token means different symbol
        self.mt5Controller = MT5Controller()
        self.SYBMOLS = ['USDJPY', 'AUDUSD']
        self.STRATEGIES = [SwingScalping]
        self.STRATEGIES_RUNNING = []
        self.tg_available = False
        self.definedStrategy = None # defined strategy

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        optionList = []
        for strategy in self.STRATEGIES:
            optionList.append(InlineKeyboardButton(strategy.__name__, callback_data=strategy.__name__))
        keyboard = [
            optionList
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text("Please choose:", reply_markup=reply_markup)

    async def assignSymbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        self.selectedSymbol = query.data

    async def assignStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query

        # CallbackQueries need to be answered, even if no notification to the user is needed
        await query.answer()

        for strategy in self.STRATEGIES:
            if strategy.__name__ == query.data:
                optionList = []

                for symbol in self.SYBMOLS:
                    optionList.append(InlineKeyboardButton(symbol, callback_data=symbol))
                keyboard = [
                    optionList
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text("Please choose:", reply_markup=reply_markup)
                self.application.add_handler(CallbackQueryHandler(self.assignSymbol))

                self.definedStrategy = strategy(self.mt5Controller, self.selectedSymbol)

        await query.edit_message_text(text=f"Strategy is defined: {query.data}")

    async def runStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.definedStrategy:
            self.definedStrategy.run()
            await update.message.reply_text(f"{self.definedStrategy.__name__} is running... ")

    def run(self):
        self.application.add_handler(CommandHandler('s', self.start))
        self.application.add_handler(CallbackQueryHandler(self.assignStrategy))
        # hander run the strategy
        self.application.add_handler(CommandHandler('run', self.runStrategy))
        print('TG Running ...')
        self.application.run_polling()

tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
print()

# class Telegram_Bot:
#     def __init__(self, token):
#         self.chat_id = False
#         self.bot = telebot.TeleBot(token)  # different token means different symbol
#         self.mt5Controller = MT5Controller
#         self.SYBMOLS = ['USDJPY', 'AUDUSD']
#         self.STRATEGIES = [SwingScalping]
#         self.STRATEGIES_SET = []
#         self.tg_available = False
#
#     def getStrategyListText(self):
#         txt = ''
#         for i, strategy in enumerate(self.STRATEGIES):
#             txt += f"{i + 1}.: {strategy.__name__}\n"
#         return txt
#
#     def run(self):
#         @self.bot.message_handler(commands=['start'])
#         def startTrade(message):
#             # ask the strategy to be choose
#             strategyListText = self.getStrategyListText()
#             responseMsg = self.bot.reply_to(message, f"{strategyListText}\nPlease select the strategy. ")
#             self.bot.register_next_step_handler(responseMsg, selectStrategy)
#
#         def selectStrategy(message):
#             strategyIndex = message.text
#             if not strategyIndex.isdigit():
#                 self.bot.send_message(message.chat.id, "This is not a number")
#                 return False
#             strategyClass = self.STRATEGIES[int(strategyIndex) - 1]
#             self.bot.send_message(message.chat.id, f"You have selected {strategyClass.__name__} strategy. ")
#             self.STRATEGIES_SET.append(strategyClass(self.mt5Controller, 'AUDUSD'))
#
#         @self.bot.message_handler(commands=['run'])
#         def runStrategy(message):
#             # ask the strategy to be choose
#             strategyListText = self.getStrategyListText()
#             responseMsg = self.bot.reply_to(message, f"{strategyListText}\nPlease select the strategy. ")
#             self.bot.register_next_step_handler(responseMsg, selectStrategy)
#
#         @self.bot.message_handler(commands=['long'])
#         def longPosition(message):
#             pass
#
#         @self.bot.message_handler(commands=['short'])
#         def shortPosition(message):
#             pass
#
#         @self.bot.message_handler(commands=['status'])
#         def getAccountStatus(message):
#             pass
#
#         self.bot.polling()
