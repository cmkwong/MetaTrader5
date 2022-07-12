from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from TkWindow import TkWindow
from mt5f.MT5Controller import MT5Controller

class MT5ControlPage(TkWindow):
    def __init__(self):
        super(MT5ControlPage, self).__init__()
        self.isSetParam = False

    def openControl(self, parent):
        self.root = Toplevel(parent)
        self.createFrame(self.root, 'MT5 Control', {
            ''
        })
        if not self.isSetParam:
            self.openInputParam(parent)

    def openInputParam(self, parent):
        self.settingWindow = Toplevel(parent)
        Label(self.settingWindow, text="Local Data Path").grid(row=0, column=0)
        self.e_dataPath = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0, column=1)  # local data path
        Label(self.settingWindow, text="Timezone").grid(row=0, column=0)
        self.e_timezone = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, 'Hongkong').grid(row=1, column=1)
        Label(self.settingWindow, text="Deposit").grid(row=0, column=0)
        self.e_deposit = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, 'USD').grid(row=2, column=1)
        Label(self.settingWindow, text="Type Filling").grid(row=0, column=0)
        self.e_typeFilling = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, 'ioc').grid(row=3, column=1)

        self.okBtn = Button(self.settingWindow, text='RUN', command=self.setParam).grid(row=3, column=0, columnspan=2)

    def setParam(self):
        self.dataPath = self.e_dataPath.get()
        self.timezone = self.e_timezone.get()
        self.deposit = self.e_deposit.get()
        self.typeFilling = self.e_typeFilling.get()
        self.mt5Controller = MT5Controller(self.dataPath, self.timezone, self.deposit, self.typeFilling)
        self.settingWindow.destroy()
        # popup control panel

    def run(self):
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

