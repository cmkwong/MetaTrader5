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


class MT5ControlWindow(TkWindow):
    def __init__(self):
        super(MT5ControlWindow, self).__init__()
        self.isSetParam = False
        # variables
        self.var_timezone = StringVar()
        self.var_ccy = StringVar()
        self.var_typeFilling = StringVar()

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
                         command=lambda: self.openWindow(root, [self.getGetDataFrame])),
            TkInitWidget(cat=cat, id='uploadData', type=self.BUTTON, label='Upload DB', pos=(1, 1, 1), command=None),
            TkInitWidget(cat=cat, id='Setting', type=self.BUTTON, label='Setting', pos=(2, 1, 1),
                         command=lambda: self.openWindow(root, [self.getInputParamFrame])),
        ], "Operation Panel")
        return frame

    def onClickSettingOk(self, root, cat):
        allParamFilled = True
        params = []
        for id, widget in self.widgets[cat].items():
            if widget.widgetName == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            allParamFilled = allParamFilled and bool(self.getWidgetValue(cat, id))
            params.append(param)
        if (allParamFilled):
            self.mt5Controller = MT5Controller(*params)
            self.isSetParam = True
            print('Param set')
            root.destroy()

    def getInputParamFrame(self, root):
        cat = MT5Controller.__name__
        initWidgets = self.get_params_initWidgets(MT5Controller, cat)
        # append save button
        initWidgets.append(TkInitWidget(cat=cat, id='saveSetting', type=self.BUTTON, label="Save", command=lambda: self.onClickSettingOk(root, cat), pos=(len(initWidgets), 0, 1)))
        frame = self.createFrame(root, initWidgets, 'MT5Controller')
        # frame = self.createFrame(window, [
        #     InitWidget(id='dataPath', type=self.TEXTFIELD, label='Local Data Path',
        #                default='C:/Users/Chris/projects/210215_mt5/docs',
        #                pos=(0, 0, 1), style={'width': 50, 'borderwidth': 3}),
        #     InitWidget(id='timezone', type=self.DROPDOWN, var=self.var_timezone, label='TimeZone',
        #                value=['Hongkong', 'en_US'], default='Hongkong',
        #                pos=(1, 0, 1), style={'width': 50}),
        #     InitWidget(id='deposit', type=self.DROPDOWN, var=self.var_ccy, label='Deposit CCY',
        #                value=['USD', 'GBP', 'EUR'], default='USD',
        #                pos=(2, 0, 1), style={'width': 50}),
        #     InitWidget(id='typeFilling', type=self.DROPDOWN, var=self.var_typeFilling, label='Type Filling',
        #                value=['ioc', 'fok', 'return'], default='ioc',
        #                pos=(3, 0, 1), style={'width': 50}),
        #     InitWidget(id='saveSetting', type=self.BUTTON, label="Save", command=lambda: self.onClickSettingOk(window),
        #                pos=(4, 0, 1))
        # ], 'MT5 Setting')
        return frame

    def getGetDataFrame(self, root):
        cat = 'getData'
        initWidgets = self.get_params_initWidgets(self.mt5Controller.mt5PricesLoader.get_data, cat)
        frame = self.createFrame(root, initWidgets, 'Get Data Setting')
        # frame = self.createFrame(root, [
        #     InitWidget(cat=cat, id='symbols', type=self.TEXTFIELD, label='Symbols', default='USDJPY', pos=(0, 0, 1)),
        #     InitWidget(cat=cat, id='start', type=self.CALENDAR, label='Start', default='2010-01-01', pos=(1, 0, 1)),
        #     InitWidget(cat=cat, id='end', type=self.CALENDAR, label='End', default='2022-01-01', pos=(2, 0, 1)),
        #     InitWidget(cat=cat, id='timeframe', type=self.DROPDOWN, label='Time Frame',
        #                value=list(timeModel.timeframe_ftext_dicts.keys()), pos=(3, 0, 1)),
        #
        # ], 'Get Data Setting')
        return frame

    def run2(self):
        self.controlWindow = Toplevel()

        Label(self.settingWindow, text="Symbols").grid(row=0, column=0)
        self.e_symbols = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)

        Label(self.settingWindow, text="Start").grid(row=0, column=0)
        self.e_start = Calendar(self.settingWindow, selectmode='day', year=2010, month=1, day=1).grid(row=0, column=1)

        Calendar(self.settingWindow, text="End").grid(row=0, column=0)
        self.e_end = Calendar(self.settingWindow, selectmode='day', year=2022, month=1, day=1).grid(row=0, column=1)

        Label(self.settingWindow, text="Timeframe").grid(row=0, column=0)
        self.e_timeframe = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)

        Label(self.settingWindow, text="local").grid(row=0, column=0)
        self.e_local = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)

        Label(self.settingWindow, text="Latest").grid(row=0, column=0)
        self.e_latest = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)

        Label(self.settingWindow, text="Count").grid(row=0, column=0)
        self.e_count = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)
