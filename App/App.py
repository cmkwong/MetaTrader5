from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from TkWindow import TkWindow
from mt5f.MT5Controller import MT5Controller


class MainPage(TkWindow):
    def __init__(self):
        self.root = Tk()
        super().__init__(self.root)
        self.root.title('Forex App')
        self.root.geometry("400x100")
        self.operations = ['MT5', 'Data', 'Strategies', 'Setting']
        self.operationSelected = StringVar()

    def run(self):
        # set default variable
        self.operationSelected.set(self.operations[0])

        # define element
        operationFrame = self.createFrame("Operation Selection", {
            'operationDropdown': {
                'wtype': 'OptionMenu',
                'label': "Please select the operation",
                'value': self.operations,
                'variable': self.operationSelected,
                "pos": (0, 0, 1)
            },
            "operationSubmit": {
                'wtype': "Button",
                "label": "Submit",
                "value": self.onOperationClicked,
                "pos": (0, 1, 1)
            },
            "operationStatus": {
                "wtype": "Label",
                "label": "Now the operation is running: ",
                "value": "",
                "pos": (1, 0, 2)
            }
        })

        operationFrame.pack()

        # widget defined
        # self.label = Label(self.root, text=self.operations[0])
        # self.dropDown = OptionMenu(self.root, self.operationSelected, *self.operations)
        # self.okBtn = Button(self.root, text="Open", command=self.onOperationClicked)

        # interface display
        # self.label.grid(row=1, column=0, columnspan=2)
        # self.dropDown.grid(row=0, column=0)
        # self.okBtn.grid(row=0, column=1)

    def onOperationClicked(self):
        operation = self.operationSelected.get()
        print(operation)
        self.widgets['operationStatus']['text'] = operation
        # if operation == "MT5":
        #     MT5Page()

class MT5Page:
    def inputParam(self):
        self.settingWindow = Toplevel()
        Label(self.settingWindow, text="Local Data Path").grid(row=0, column=0)
        self.e_dataPath = Entry(self.settingWindow, width=50, borderwidth=3).insert(0, '').grid(row=0,
                                                                                                column=1)  # local data path
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


class App(AppSetting):
    def __init__(self):
        AppSetting.__init__(self)
        self.MainPage = MainPage()

    def run(self):
        self.MainPage.run()
        self.MainPage.root.mainloop()


app = App()
app.run()
