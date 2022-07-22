from tkinter import *

from TkInitWidget import TkInitWidget
from TkWindow import TkWindow

from backtest import timeModel
# Atom
from myUtils import paramModel


class WindowMT5PricesLoader(TkWindow):
    def __init__(self, mt5Controller):
        super(WindowMT5PricesLoader, self).__init__()
        self.isSetParam = False
        self.mt5Controller = mt5Controller
        self.getDataCount = 1

    def run(self, root):
        self.openTopWindow(root, [self.getGetDataFrame, self.getStatusFrame], '400x600')

    def onClickGetData(self, root, cat):
        params = []
        for id, widget in self.widgets[cat].items():
            if type(widget).__name__ == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            params.append(param)
        requiredParams = paramModel.insert_params(self.mt5Controller.mt5PricesLoader.getPrices, params)
        Prices = self.mt5Controller.mt5PricesLoader.getPrices(**requiredParams)
        # show the status
        cols = [k for k in Prices.__dataclass_fields__.keys()]
        text = f"""
        Data got times {self.getDataCount}:
        The line of row: {len(Prices.c)}
        The columns: {len(cols)}
        """
        self.widgets['status']['showStatus'].insert(END, text)
        self.widgets['status']['showStatus'].see(END)
        self.getDataCount += 1

    def getGetDataFrame(self, root):
        cat = 'getData'
        # initWidgets = self.get_params_initWidgets(self.mt5Controller.mt5PricesLoader.get_data, cat)
        # frame = self.createFrame(root, initWidgets, 'Get Data Setting')
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='symbols', type=self.TEXTFIELD, label='Symbols', default='USDJPY EURUSD', pos=(0, 0, 1), targetType=list),
            TkInitWidget(cat=cat, id='start', type=self.CALENDAR, label='Start', default='2011-01-03', pos=(1, 0, 1)),
            TkInitWidget(cat=cat, id='end', type=self.CALENDAR, label='End', default='2022-01-01', pos=(2, 0, 1)),
            TkInitWidget(cat=cat, id='timeframe', type=self.DROPDOWN, label='Time Frame',
                         value=list(timeModel.timeframe_ftext_dicts.keys()), default='1H', pos=(3, 0, 1)),
            TkInitWidget(cat=cat, id='latest', type=self.DROPDOWN, label='latest', value=[0, 1], default='0', pos=(4, 0, 1)),
            TkInitWidget(cat=cat, id='count', type=self.TEXTFIELD, label='Count', default=10, pos=(5, 0, 1)),
            TkInitWidget(cat=cat, id='ohlcvs', type=self.TEXTFIELD, label='ohlcvs', default='111111', pos=(6, 0, 1)),
            TkInitWidget(cat=cat, id='submit', type=self.BUTTON, label='Submit', command=lambda: self.onClickGetData(root, cat), pos=(7, 0, 1))
        ], 'Get Data Setting')
        return frame

    def getStatusFrame(self, root):
        cat = 'status'
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='showStatus', type=self.SCROLLEDTEXT, pos=(0, 0, 1), style={'height': 10, 'width': 40})
        ])
        return frame
