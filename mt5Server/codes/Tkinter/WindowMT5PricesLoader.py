from tkinter import *

from TkInitWidget import TkInitWidget
from TkWindow import TkWindow

from mt5Server.codes.AppStorage import AppData
from mt5Server.codes.AppStorage import AppClasses
from mt5Server.codes.backtest import timeModel
# Atom
from myUtils import paramModel


class WindowMT5PricesLoader(TkWindow):
    def __init__(self):
        super(WindowMT5PricesLoader, self).__init__()
        self.getDataCount = 1

    def run(self, root):
        self.openTopWindowByFrame(root, [self.pricesFrame, self.statusFrame], title='Prices Loader', windowSize='400x600')

    def storeData(self, symbols, Prices):
        dfs = Prices.getOhlcvsFromPrices(symbols)
        for symbol, df in dfs.items():
            count = int(self.getWidgetValue('getData', 'count')) # convert into integer
            if (count > 0):
                AppData['live'][symbol] = df
            elif (count == 0):
                AppData['history'][symbol] = df

    def onClickGetData(self, root, cat):
        params = []
        for id, widget in self.widgets[cat].items():
            if type(widget).__name__ == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            params.append(param)
        requiredParams = paramModel.insert_params(AppClasses['MT5Controller'].mt5PricesLoader.getPrices, params)
        Prices = AppClasses['MT5Controller'].mt5PricesLoader.getPrices(**requiredParams)
        # store Data
        self.storeData(requiredParams['symbols'], Prices)
        # which of fields getting valid
        cols = Prices.getValidCols()
        # show the status
        text = f"""
        Data got times {self.getDataCount}:
        The line of row: {len(Prices.c)}
        The columns: {len(cols)}
        """
        self.widgets['status']['showStatus'].insert(END, text)
        self.widgets['status']['showStatus'].see(END)
        self.getDataCount += 1

    def pricesFrame(self, root):
        cat = 'getData'
        defaultSymbols = 'AUDJPY AUDCAD CADJPY AUDUSD EURGBP EURCAD EURUSD EURAUD GBPUSD USDJPY USDCAD'
        # initWidgets = self.get_params_initWidgets(self.mt5Controller.mt5PricesLoader.get_data, cat)
        # frame = self.createFrame(root, initWidgets, 'Get Data Setting')
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='symbols', type=self.TEXTFIELD, label='Symbols', default=defaultSymbols, pos=(0, 0, 1), targetType=list),
            TkInitWidget(cat=cat, id='start', type=self.CALENDAR, label='Start', default='2011-01-03', pos=(1, 0, 1)),
            TkInitWidget(cat=cat, id='end', type=self.CALENDAR, label='End', default='2022-01-01', pos=(2, 0, 1)),
            TkInitWidget(cat=cat, id='timeframe', type=self.DROPDOWN, label='Time Frame',
                         value=list(timeModel.timeframe_ftext_dicts.keys()), default='1H', pos=(3, 0, 1)),
            TkInitWidget(cat=cat, id='count', type=self.TEXTFIELD, label='Count', default='0', pos=(4, 0, 1)),
            TkInitWidget(cat=cat, id='ohlcvs', type=self.TEXTFIELD, label='ohlcvs', default='111111', pos=(5, 0, 1)),
            TkInitWidget(cat=cat, id='submit', type=self.BUTTON, label='Submit', onClick=lambda: self.onClickGetData(root, cat), pos=(6, 0, 1))
        ], 'Get Data Setting')
        return frame

    def statusFrame(self, root):
        cat = 'status'
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='showStatus', type=self.SCROLLEDTEXT, pos=(0, 0, 1), style={'height': 10, 'width': 40})
        ])
        return frame
