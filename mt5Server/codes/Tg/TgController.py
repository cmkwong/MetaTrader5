from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
import threading
import asyncio

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.Scalping.SwingScalping import SwingScalping
from mt5Server.codes import config

SET_SYMBOL, SET_STRATEGY = map(chr, range(2))
END = map(chr, range(2, 3))


class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.application = Application.builder().token(token).build()  # different token means different symbol
        self.mt5Controller = MT5Controller()
        self.SYBMOLS = ['USDJPY', 'AUDUSD']
        self.STRATEGIES = [SwingScalping]
        self.STRATEGIES_RUNNING = []
        self.tg_available = False
        self.idleStrategies = {}  # idle strategy: {strategy_name: class object}
        self.runningStrategies = {} # running strategy: {strategy_name: class object}

    async def addStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        optionList = []
        for strategy in self.STRATEGIES:
            optionList.append(InlineKeyboardButton(strategy.__name__, callback_data=strategy.__name__))
        reply_markup = InlineKeyboardMarkup([optionList])

        await update.message.reply_text("Please choose:", reply_markup=reply_markup)

        return SET_SYMBOL

    # async def _addStrategy(self, update, context):
    #     query = update.callback_query
    #
    #     # waiting answer
    #     await query.answer()
    #
    #     # store the strategy name
    #     context.user_data['strategy'] = query.data

    async def setStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query

        # CallbackQueries need to be answered, even if no notification to the user is needed
        await query.answer()

        symbol = query.data

        definedStrategy = None
        for strategy in self.STRATEGIES:
            if strategy.__name__ == context.user_data['strategy']:
                definedStrategy = strategy(self.mt5Controller, symbol)
                self.idleStrategies[definedStrategy.getName] = definedStrategy

        await query.edit_message_text(text=f"Strategy is defined: {definedStrategy}")

        return END

    async def runStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query

        await query.answer()

        strategyName = query.data

        self.idleStrategies[strategyName].run()
        await update.message.reply_text(f"{strategyName} is running... ")

    async def endConv(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Completed.")
        return END

    async def setSymbol(self, update, context):
        optionList = []
        for symbol in self.SYBMOLS:
            optionList.append(InlineKeyboardButton(symbol, callback_data=symbol))
        reply_markup = InlineKeyboardMarkup([optionList])

        query = update.callback_query

        await query.answer()
        context.user_data['strategy'] = query.data

        await update.callback_query.edit_message_text("Please choose a symbol:", reply_markup=reply_markup)

        return SET_STRATEGY

    def run(self):
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('add', self.addStrategy)],
            states={
                END: [CommandHandler('add', self.addStrategy)],
                SET_SYMBOL: [CallbackQueryHandler(self.setSymbol)],
                SET_STRATEGY: [CallbackQueryHandler(self.setStrategy)]
            },
            fallbacks=[CommandHandler('end', self.endConv)]
        )
        print('TG Running ...')
        self.application.add_handler(conv_handler)
        self.application.run_polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
tg.run()
print()
