from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from appVariable import AppStorage, AppClasses


class WindowDataController(TkWindow):

    def run(self, root):
        self.openTopWindowByFrame(root, [self.uploadDataFrame], title="Data Controller", windowSize='1000x400')

    def uploadDataFrame(self, root):
        cat = 'data'

        historicalSymbols = list(AppStorage['history'].keys())
        liveSymbols = list(AppStorage['live'].keys())

        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='historyUpload', type=self.DROPDOWN,
                         label='Historical data: ', value=historicalSymbols, pos=(0, 0, 1)),
            TkInitWidget(cat=cat, id='liveUpload', type=self.DROPDOWN,
                         label='Live data: ', value=liveSymbols, pos=(1, 0, 1)),
            TkInitWidget(cat=cat, id='uploadForex', type=self.BUTTON,
                         label='Upload Forex Data', onClick=None, pos=(2, 0, 1)),
        ], "Data Operation")

        return frame
