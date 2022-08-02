from AppSetting import AppSetting
from TkInitWidget import TkInitWidget
from TkWindow import TkWindow
from WindowMT5PricesLoader import WindowMT5PricesLoader

from AppStorage import AppData
from mt5f.MT5Controller import MT5Controller
from backtest import timeModel

from myUtils import paramModel


class WindowMT5Controller(TkWindow):
    def __init__(self):
        super(WindowMT5Controller, self).__init__()
        self.windowMT5PricesLoader = WindowMT5PricesLoader()

    def run(self, root):
        print('param already set')
        controlWindow = self.openTopWindowByFrame(root, [self.getControlFrame], 'MT5 Control Panel')

    def getControlFrame(self, root):
        cat = "mt5Control"
        # upload/get data
        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='getData', type=self.BUTTON, label='Get Data From MT5', pos=(0, 1, 1),
                         onClick=lambda: self.windowMT5PricesLoader.run(root)),
            TkInitWidget(cat=cat, id='execute', type=self.BUTTON, label='Execute on MT5', pos=(1, 1, 1), onClick=None),
        ], "Operation Panel")
        return frame
