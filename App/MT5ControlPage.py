from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from common import InitWidget
from TkWindow import TkWindow
from mt5f.MT5Controller import MT5Controller


class MT5ControlPage(TkWindow):
    def __init__(self):
        super(MT5ControlPage, self).__init__()
        self.isSetParam = False
        # variables
        self.var_dataPath = StringVar()
        self.var_timezone = StringVar()
        self.var_ccy = StringVar()
        self.var_typeFilling = StringVar()

    def run(self, parent):
        if not self.isSetParam:
            settingWindow = self.openWindow(parent, [self.getInputParamFrame], "400x400")
            # settingWindow = Toplevel(parent)
            # settingWindow.geometry("400x400")
            # frame = self.getInputParamFrame(settingWindow)
            # frame.pack()
        else:
            print('param already set')
            # self.root = Toplevel(parent)
            # self.root.geometry("400x400")
            window = self.openWindow(parent, [self.getControlFrame], "400x400")

    def getControlFrame(self, window):
        # upload/get data
        frame = self.createFrame(window, "Operation Panel", [
            InitWidget(id='getData', type=self.BUTTON, label='Get Data From MT5', pos=(0, 1, 1), command=None),
        ])
        return frame

    def onClickSettingOk(self, window):
        print('Run Clicked')
        self.isSetParam = True
        window.destroy()

    def getInputParamFrame(self, window):
        frame = self.createFrame(window, 'MT5 Setting', [
            InitWidget(id='entry_dataPath', type=self.TEXTFIELD, label='Local Data Path',
                       pos=(0, 0, 1), style={'width': 50, 'borderwidth': 3}),
            InitWidget(id='entry_timezone', type=self.DROPDOWN, var=self.var_timezone, label='TimeZone',
                       value=['Hongkong', 'en_US'],
                       pos=(1, 0, 1), style={'width': 50}),
            InitWidget(id='entry_deposit', type=self.DROPDOWN, var=self.var_ccy, label='Deposit CCY',
                       value=['USD', 'GBP', 'EUR'],
                       pos=(2, 0, 1), style={'width': 50}),
            InitWidget(id='entry_typeFilling', type=self.DROPDOWN, var=self.var_typeFilling, label='Type Filling',
                       value=['ioc', 'fok', 'return'],
                       pos=(3, 0, 1), style={'width': 50}),
            InitWidget(id='runBtn', type=self.BUTTON, label="Run", command=lambda: self.onClickSettingOk(window),
                       pos=(4, 0, 1))
        ])
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
