from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from TkInitWidget import TkInitWidget
from TkWindow import TkWindow
from mt5f.MT5Controller import MT5Controller
from mt5f.loader.MT5PricesLoader import MT5PricesLoader
from backtest import timeModel

from utils import paramModel

class MT5ControllerWindow(TkWindow):
    def __init__(self):
        super(MT5ControllerWindow, self).__init__()
        self.isSetParam = False

    def run(self, root):
        if not self.isSetParam:
            settingWindow = self.openWindow(root, [self.getInputParamFrame])
        else:
            print('param already set')
            controlWindow = self.openWindow(root, [self.getControlFrame])

    def getControlFrame(self, root):
        cat = "mt5Control"
        # upload/get data
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='getData', type=self.BUTTON, label='Get Data From MT5', pos=(0, 1, 1),
                         command=lambda: self.openWindow(root, [self.getGetDataFrame], '400x800')),
            TkInitWidget(cat=cat, id='uploadData', type=self.BUTTON, label='Upload DB', pos=(1, 1, 1), command=None),
            TkInitWidget(cat=cat, id='Setting', type=self.BUTTON, label='Setting', pos=(2, 1, 1),
                         command=lambda: self.openWindow(root, [self.getInputParamFrame])),
        ], "Operation Panel")
        return frame

    def onClickSettingOk(self, root, cat):
        allParamFilled = True
        params = []
        for id, widget in self.widgets[cat].items():
            if type(widget).__name__ == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            allParamFilled = allParamFilled and bool(self.getWidgetValue(cat, id))
            params.append(param)
        if (allParamFilled):
            self.mt5Controller = MT5Controller(*params)
            self.isSetParam = True
            print('Param set')
            root.destroy()

    def onClickGetData(self, root, cat):
        params = []
        for id, widget in self.widgets[cat].items():
            if widget.widgetName == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            params.append(param)
        requiredParams = paramModel.insert_params(self.mt5Controller.mt5PricesLoader.get_data, params)


    def getInputParamFrame(self, root):
        cat = MT5Controller.__name__
        initWidgets = self.get_params_initWidgets(MT5Controller, cat)
        # append save button
        initWidgets.append(TkInitWidget(cat=cat, id='saveSetting', type=self.BUTTON, label="Save", command=lambda: self.onClickSettingOk(root, cat), pos=(len(initWidgets), 0, 1)))
        frame = self.createFrame(root, initWidgets, 'MT5Controller')
        return frame

    def getGetDataFrame(self, root):
        cat = 'getData'
        # initWidgets = self.get_params_initWidgets(self.mt5Controller.mt5PricesLoader.get_data, cat)
        # frame = self.createFrame(root, initWidgets, 'Get Data Setting')
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='symbols', type=self.TEXTFIELD, label='Symbols', default='USDJPY EURUSD', pos=(0, 0, 1), targetType=list),
            TkInitWidget(cat=cat, id='start', type=self.CALENDAR, label='Start', default='2010-01-01', pos=(1, 0, 1)),
            TkInitWidget(cat=cat, id='end', type=self.CALENDAR, label='End', default='2022-01-01', pos=(2, 0, 1)),
            TkInitWidget(cat=cat, id='timeframe', type=self.DROPDOWN, label='Time Frame',
                         value=list(timeModel.timeframe_ftext_dicts.keys()), default='1H', pos=(3, 0, 1)),
            TkInitWidget(cat=cat, id='local', type=self.DROPDOWN, label='Local', value=[0, 1], default='0', pos=(4, 0, 1)),
            TkInitWidget(cat=cat, id='latest', type=self.DROPDOWN, label='latest', value=[0, 1], default='0', pos=(5, 0, 1)),
            TkInitWidget(cat=cat, id='count', type=self.TEXTFIELD, label='Count', default=10, pos=(6, 0, 1)),
            TkInitWidget(cat=cat, id='submit', type=self.BUTTON, label='Submit', command=lambda: self.onClickGetData(root, cat), pos=(7, 0, 1))
        ], 'Get Data Setting')
        return frame
