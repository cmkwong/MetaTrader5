from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler, InlineQueryHandler, MessageHandler, filters
import threading
import asyncio

from mt5Server.codes.Mt5f.MT5Controller import MT5Controller
from mt5Server.codes.Strategies.Scalping.SwingScalping import SwingScalping
from mt5Server.codes import config

SET_SYMBOL, SET_STRATEGY = map(chr, range(2))
SELECT, RUN, END = map(chr, range(2, 5))
CHATID = '1051403979'

class Telegram_Bot:
    def __init__(self, token):
        self.chat_id = False
        self.application = Application.builder().token(token).arbitrary_callback_data(True).build()  # different token means different symbol
        self.mt5Controller = MT5Controller()
        self.SYBMOLS = ['USDJPY', 'USDCAD', 'AUDJPY', 'AUDUSD']
        self.STRATEGIES = [SwingScalping]
        self.STRATEGIES_RUNNING = []
        self.tg_available = False
        self.idleStrategies = {}  # idle strategy: {strategy_name: class object}
        self.runningStrategies = {}  # running strategy: {strategy_name: class object}
        self.targetStrategy = None

    async def listStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        optionList = []
        for strategy in self.STRATEGIES:
            optionList.append(InlineKeyboardButton(strategy.__name__, callback_data=strategy.__name__))
        reply_markup = InlineKeyboardMarkup([optionList])

        await update.message.reply_text("Please choose:", reply_markup=reply_markup)

        return SET_SYMBOL

    async def setStrategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query

        # CallbackQueries need to be answered, even if no notification to the user is needed
        await query.answer()

        symbol = query.data

        definedStrategy = None
        for strategy in self.STRATEGIES:
            if strategy.__name__ == context.user_data['strategy']:
                definedStrategy = strategy(self.mt5Controller, symbol, tg=self)
                self.idleStrategies[definedStrategy.getName] = definedStrategy

        await query.edit_message_text(text=f"Strategy is defined: {definedStrategy}")

        return END

    async def listIdleStrategy(self, update, context):
        if self.idleStrategies:
            optionList = []
            for strategyName in self.idleStrategies.keys():
                optionList.append(InlineKeyboardButton(strategyName, callback_data=strategyName))
            reply_markup = InlineKeyboardMarkup([optionList])

            await update.message.reply_text("Please choose:", reply_markup=reply_markup)

            return SELECT
        else:
            await update.message.reply_text('There is no idle strategy yet. ')
            return END

    async def selectStrategy(self, update, context):
        query = update.callback_query

        await query.answer()

        strategyName = query.data

        # thread = threading.Thread(target=self.idleStrategies[strategyName].run, args=[update, context])
        # thread.start()

        # status = self.idleStrategies[strategyName].run()

        # add the strategy in running strategy
        self.targetStrategy = self.idleStrategies[strategyName]

        await self.targetStrategy.run(update, context)

        return RUN

    async def endConv(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Completed.")
        return END

    async def listSymbol(self, update, context):
        optionList = []
        for symbol in self.SYBMOLS:
            optionList.append(InlineKeyboardButton(symbol, callback_data=symbol))
        reply_markup = InlineKeyboardMarkup([optionList])

        query = update.callback_query

        await query.answer()
        context.user_data['strategy'] = query.data

        await update.callback_query.edit_message_text("Please choose a symbol:", reply_markup=reply_markup)

        return SET_STRATEGY

    # async def preActingNotice(self, update, context, msg):
    #     await update.message.send_message(text=msg, chat_id=CHATID)

    async def start(self, update, context):
        self.chat_id = update.effective_chat.id
        await update.effective_message.reply_html(
            f"Your chat id is <code>{update.effective_chat.id}</code>."
        )

    def run(self):
        addConv = ConversationHandler(
            entry_points=[CommandHandler('add', self.listStrategy)],
            states={
                SET_SYMBOL: [CallbackQueryHandler(self.listSymbol)],
                SET_STRATEGY: [CallbackQueryHandler(self.setStrategy)],
                END: [CommandHandler('add', self.listStrategy)]
            },
            fallbacks=[CommandHandler('end', self.endConv)]
        )
        runConv = ConversationHandler(
            entry_points=[CommandHandler('run', self.listIdleStrategy)],
            states={
                SELECT: [CallbackQueryHandler(self.selectStrategy)],
                # RUN: [CallbackQueryHandler(self.targetStrategy.run)],
                END: [CommandHandler('run', self.listIdleStrategy)]
            },
            fallbacks=[]
        )
        print('TG Running ...')
        self.application.add_handler(addConv)
        self.application.add_handler(runConv)
        # show chat id
        self.application.add_handler(CommandHandler('start', self.start))
        self.application.run_polling()


tg = Telegram_Bot('5647603910:AAHsqwx7YGoDRicWAhEE4TWi1vk5zN69Fl4')
# tg.preActingNotice('hello', 0.1, 0.1, 0.1, 0.1)
tg.run()
print()
