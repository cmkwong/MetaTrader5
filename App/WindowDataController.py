from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from AppStorage import AppData, AppClasses
from data import DataController

class WindowDataController(TkWindow):

    def run(self, root):
        self.openTopWindowByFrame(root, [self.uploadDataFrame], title="Data Controller", windowSize='1000x400')

    def uploadDataFrame(self, root):
        cat = 'data'

        historicalSymbols = list(AppData['history'].keys())
        liveSymbols = list(AppData['live'].keys())
        if len(historicalSymbols) == 0: historicalSymbols.append('No Data')
        if len(liveSymbols) == 0: liveSymbols.append('No Data')

        frame = self.createFrame(root, [
            TkInitWidget(cat=cat, id='tableName', type=self.TEXTFIELD,
                         label='Table Name'),
            TkInitWidget(cat=cat, id='historyUpload', type=self.DROPDOWN,
                         label='Historical data: ', value=historicalSymbols),
            TkInitWidget(cat=cat, id='btn_historyUpload', type=self.BUTTON,
                         label='Upload', onClick=None),
            TkInitWidget(cat=cat, id='liveUpload', type=self.DROPDOWN,
                         label='Live data: ', value=liveSymbols),
            TkInitWidget(cat=cat, id='btn_liveUpload', type=self.BUTTON,
                         label='Upload', onClick=None),
        ], "Data Operation")
        return frame

    def onUploadHistoryData(self, cat):
        self.getWidgetValue(cat, historyUpload)
        AppClasses[DataController.__name__].uploadForexData(AppData[''])
