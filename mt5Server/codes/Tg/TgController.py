import telebot
import threading
import asyncio

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.Scalping.SwingScalping import SwingScalping
from mt5Server.codes import config


class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.bot = telebot.TeleBot(token)  # different token means different symbol
        self.mt5Controller = MT5Controller
        self.SYBMOLS = ['USDJPY', 'AUDUSD']
        self.STRATEGIES = [SwingScalping]
        self.STRATEGIES_SET = []
        self.tg_available = False

    def getStrategyListText(self):
        txt = ''
        for i, strategy in enumerate(self.STRATEGIES):
            txt += f"{i + 1}.: {strategy.__name__}\n"
        return txt

    def run(self):
        @self.bot.message_handler(commands=['start'])
        def startTrade(message):
            # ask the strategy to be choose
            strategyListText = self.getStrategyListText()
            responseMsg = self.bot.reply_to(message, f"{strategyListText}\nPlease select the strategy. ")
            self.bot.register_next_step_handler(responseMsg, selectStrategy)

        def selectStrategy(message):
            strategyIndex = message.text
            if not strategyIndex.isdigit():
                self.bot.send_message(message.chat.id, "This is not a number")
                return False
            strategyClass = self.STRATEGIES[int(strategyIndex) - 1]
            self.bot.send_message(message.chat.id, f"You have selected {strategyClass.__name__} strategy. ")
            self.STRATEGIES_SET.append(strategyClass(self.mt5Controller, 'AUDUSD'))

        @self.bot.message_handler(commands=['run'])
        def runStrategy(message):
            # ask the strategy to be choose
            strategyListText = self.getStrategyListText()
            responseMsg = self.bot.reply_to(message, f"{strategyListText}\nPlease select the strategy. ")
            self.bot.register_next_step_handler(responseMsg, selectStrategy)

        @self.bot.message_handler(commands=['long'])
        def longPosition(message):
            pass

        @self.bot.message_handler(commands=['short'])
        def shortPosition(message):
            pass

        @self.bot.message_handler(commands=['status'])
        def getAccountStatus(message):
            pass

        self.bot.polling()
