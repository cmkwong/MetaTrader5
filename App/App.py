import tkinter as tk

from AppStorage import AppClasses
from AppSetting import AppSetting
from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from WindowMT5Controller import WindowMT5Controller
from WindowDataController import WindowDataController
from mt5f.MT5Controller import MT5Controller
from data.ServerConnector import ServerConnector


class MainPage(TkWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        self.windowMT5Controller = WindowMT5Controller()
        self.windowDataController = WindowDataController()

    def onOperationClicked(self):
        operation = self.getWidgetValue('main', 'operationDropdown')
        print(operation)
        self.widgets['main']['operationStatus']['text'] = operation
        if operation == "MT5":
            if MT5Controller.__name__ not in AppClasses.keys():
                self.openTopWindowByFrame(self.root, [self.getInputParamFrame], 'MT5Controller Setting', classFn=MT5Controller)
            else:
                self.windowMT5Controller.run(self.root)
        elif operation == 'Data':
            if ServerConnector.__name__ not in AppClasses.keys():
                AppClasses[ServerConnector.__name__] = ServerConnector()
            self.windowDataController.run(self.root)

    def onSetting(self):
        operation = self.getWidgetValue('main', 'operationDropdown')
        if operation == 'MT5':
            self.openTopWindowByFrame(self.root, [self.getInputParamFrame], 'MT5Controller Setting', classFn=MT5Controller)

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
                         onClick=self.onOperationClicked, pos=(0, 1, 1)),
            TkInitWidget(cat=cat, id='operationStatus', type=self.LABEL,
                         label='Now the operation is running: ', pos=(1, 0, 2)),
            TkInitWidget(cat=cat, id='setting', type=self.BUTTON, label="Setting", onClick=self.onSetting, pos=(2, 0, 2))
        ], "Operation Selection")

        return frame

    def onClickSaveParam(self, root, cat, classFn):
        allParamFilled = True
        params = []
        for id, widget in self.widgets[cat].items():
            if type(widget).__name__ == self.BUTTON: continue
            param = self.getWidgetValue(cat, id)
            allParamFilled = allParamFilled and bool(self.getWidgetValue(cat, id))
            params.append(param)
        if (allParamFilled):
            AppClasses[classFn.__name__] = classFn(*params)
            print('Param set')
            root.destroy()

    def getInputParamFrame(self, root, classFn):
        cat = classFn.__name__
        initWidgets = self.get_params_initWidgets(classFn, cat)
        # append save button
        initWidgets.append(TkInitWidget(cat=cat, id='saveSetting', type=self.BUTTON, label="Save", onClick=lambda: self.onClickSaveParam(root, cat, classFn), pos=(len(initWidgets), 0, 1)))
        frame = self.createFrame(root, initWidgets, 'Setting')
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
