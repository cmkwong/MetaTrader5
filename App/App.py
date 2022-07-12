from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar
from dataclasses import dataclass
import collections

from AppSetting import AppSetting
from TkWindow import TkWindow
from mt5f.MT5Controller import MT5Controller


@dataclass
class Element:
    id: str
    type: str
    value: any = None
    var: any = None
    command: any = None
    label: str = ''
    pos: tuple = (0, 0, 0)
    style: dict = None

fields = ['id', 'type', 'value', 'var', 'command', 'label', 'pos', 'style']
ElementC = collections.namedtuple('ElementC', fields, defaults=(None,) * len(fields))

class MainPage(TkWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        self.root = Tk()
        self.root.title('Forex App')
        self.root.geometry("400x100")
        self.operations = ['MT5', 'Data', 'Strategies', 'Setting']
        self.operationSelected = StringVar()

    def onOperationClicked(self):
        operation = self.operationSelected.get()
        print(operation)
        self.widgets['operationStatus']['text'] = operation
        # if operation == "MT5":
        #     MT5Page()

    def run(self):
        # set default variable
        self.operationSelected.set(self.operations[0])

        # define element
        # operationFrame2 = self.createFrame(self.root, "Operation Selection", {
        #     'operationDropdown': {
        #         'wtype': self.DROPDOWN,
        #         'label': "Please select the operation",
        #         'value': self.operations,
        #         'variable': self.operationSelected,
        #         "pos": (0, 0, 1)
        #     },
        #     "operationSubmit": {
        #         'wtype': self.BUTTON,
        #         "label": "Submit",
        #         "value": self.onOperationClicked,
        #         "pos": (0, 1, 1)
        #     },
        #     "operationStatus": {
        #         "wtype": self.LABEL,
        #         "label": "Now the operation is running: ",
        #         "value": "",
        #         "pos": (1, 0, 2)
        #     }
        # })
        dropdown = Element('operationDropdown', type=self.DROPDOWN,
                           label="Please select the operation: ", value=self.operations,
                           var=self.operationSelected, pos=(0, 0, 1))
        btn = Element('operationSubmit', type=self.BUTTON, label="Submit",
                      command=self.onOperationClicked, pos=(0, 1, 1))
        label = Element('operationStatus', type=self.LABEL,
                        label='Now the operation is running: ', pos=(1, 0, 2))

        operationFrame = self.createFrame(self.root, "Operation Selection",
                                          [dropdown, btn, label]
                                          )

        # pack the frame
        operationFrame.pack()

class App(AppSetting):
    def __init__(self):
        AppSetting.__init__(self)
        self.MainPage = MainPage()

    def run(self):
        self.MainPage.run()
        self.MainPage.root.mainloop()


app = App()
app.run()
