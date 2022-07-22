import tkinter as tk

from AppSetting import AppSetting
from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from WindowMT5Controller import WindowMT5Controller


class MainPage(TkWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        self.windowMT5Controller = WindowMT5Controller()

    def onOperationClicked(self):
        operation = self.variables['main']['operationDropdown'].get()
        print(operation)
        self.widgets['main']['operationStatus']['text'] = operation
        if operation == "MT5":
            self.windowMT5Controller.run(self.root)
        elif operation == 'Data':
            pass

    def run(self):
        # define subpage controller
        self.root = tk.Tk()
        frame = self.getMainFrame()
        frame.pack()

    def getMainFrame(self):
        cat = 'main'

        # set default variable
        operations = ['MT5', 'Data', 'Strategies', 'Setting']

        # define element
        frame = self.createFrame(self.root, [
            TkInitWidget(cat=cat, id='operationDropdown', type=self.DROPDOWN, default=operations[0],
                         label="Please select the operation: ", value=operations,
                         pos=(0, 0, 1)),
            TkInitWidget(cat=cat, id='operationSubmit', type=self.BUTTON, label="Submit",
                         command=self.onOperationClicked, pos=(0, 1, 1)),
            TkInitWidget(cat=cat, id='operationStatus', type=self.LABEL,
                         label='Now the operation is running: ', pos=(1, 0, 2))
        ], "Operation Selection")

        return frame


class App(AppSetting):
    def __init__(self):
        AppSetting.__init__(self)
        self.MainPage = MainPage()

    def run(self):
        self.MainPage.run()
        self.MainPage.root.mainloop()


app = App()
app.run()
