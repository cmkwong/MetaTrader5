from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import filedialog
from tkcalendar import Calendar

from AppSetting import AppSetting
from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from MT5ControllerWindow import MT5ControllerWindow


class MainPage(TkWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        self.root = Tk()
        self.root.title('Forex App')
        self.root.geometry("400x100")
        # self.operationSelected = StringVar()
        # define subpage controller
        self.mt5ControlPage = MT5ControllerWindow()

    def onOperationClicked(self):
        operation = self.variables['main']['operationDropdown'].get()
        print(operation)
        self.widgets['main']['operationStatus']['text'] = operation
        if operation == "MT5":
            self.mt5ControlPage.run(self.root)

    def run(self):
        cat = 'main'

        # set default variable
        operations = ['MT5', 'Data', 'Strategies', 'Setting']

        # define element
        dropdown = TkInitWidget(cat=cat, id='operationDropdown', type=self.DROPDOWN, default=operations[0],
                                label="Please select the operation: ", value=operations,
                                pos=(0, 0, 1))
        btn = TkInitWidget(cat=cat, id='operationSubmit', type=self.BUTTON, label="Submit",
                           command=self.onOperationClicked, pos=(0, 1, 1))
        label = TkInitWidget(cat=cat, id='operationStatus', type=self.LABEL,
                             label='Now the operation is running: ', pos=(1, 0, 2))

        operationFrame = self.createFrame(self.root, [dropdown, btn, label], "Operation Selection")

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
