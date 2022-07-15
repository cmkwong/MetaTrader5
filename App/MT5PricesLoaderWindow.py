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

class MT5PricesLoaderWindow(TkWindow):
    def __init__(self):
        super(MT5PricesLoaderWindow, self).__init__()
        self.isSetParam = False

    def run(self, root):
        if not self.isSetParam:
            self.openWindow(root, [self.settingFrame], "Setting")
        else:
            pass

    def settingFrame(self, root):

        self.createFrame(root, [
            TkInitWidget()
        ])