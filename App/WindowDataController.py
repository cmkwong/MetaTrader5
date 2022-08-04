from TkWindow import TkWindow
from TkInitWidget import TkInitWidget
from AppStorage import AppData, AppClasses
from data.DataController import DataController

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
                         label='Upload', onClick=lambda: self.onUploadHistoryData(cat)),
        ], "Data Operation")
        return frame

    def onUploadHistoryData(self, cat):
        tableName = self.getWidgetValue(cat, 'tableName')
        symbol = self.getWidgetValue(cat, 'historyUpload')
        AppClasses[DataController.__name__].uploadForexData(AppData['history'][symbol], tableName=tableName)
