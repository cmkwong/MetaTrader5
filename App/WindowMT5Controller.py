from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from TkInitWidget import TkInitWidget
from TkWindow import TkWindow
from WindowMT5PricesLoader import WindowMT5PricesLoader

from mt5f.MT5Controller import MT5Controller
from backtest import timeModel

from myUtils import paramModel


class WindowMT5Controller(TkWindow):
    def __init__(self):
        super(WindowMT5Controller, self).__init__()
        self.isSetParam = False

    def run(self, root):
        if not self.isSetParam:
            settingWindow = self.openTopWindow(root, [self.getInputParamFrame])
        else:
            print('param already set')
            controlWindow = self.openTopWindow(root, [self.getControlFrame])

    def getControlFrame(self, root):
        cat = "mt5Control"
        # upload/get data
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='getData', type=self.BUTTON, label='Get Data From MT5', pos=(0, 1, 1),
                         command=lambda: self.windowMT5PricesLoader.run(root)),
            TkInitWidget(cat=cat, id='execute', type=self.BUTTON, label='Execute on MT5', pos=(1, 1, 1), command=None),
            TkInitWidget(cat=cat, id='Setting', type=self.BUTTON, label='Setting', pos=(2, 1, 1),
                         command=lambda: self.openTopWindow(root, [self.getInputParamFrame])),
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
            self.windowMT5PricesLoader = WindowMT5PricesLoader(self.mt5Controller)
            self.isSetParam = True
            print('Param set')
            root.destroy()

    def getInputParamFrame(self, root):
        cat = MT5Controller.__name__
        initWidgets = self.get_params_initWidgets(MT5Controller, cat)
        # append save button
        initWidgets.append(TkInitWidget(cat=cat, id='saveSetting', type=self.BUTTON, label="Save", command=lambda: self.onClickSettingOk(root, cat), pos=(len(initWidgets), 0, 1)))
        frame = self.createFrame(root, initWidgets, 'MT5Controller')
        return frame
