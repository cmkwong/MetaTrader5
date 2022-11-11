from myUtils.DfModel import DfModel
from mt5Server.codes.Data.NodejsServerController import NodejsServerController
from mt5Server.codes.Mt5f.MT5Controller import MT5Controller

import pandas as pd


class DataFeeder(NodejsServerController):
    def __init__(self, mt5Controller):
        super(DataFeeder, self).__init__()
        self.mT5Controller = mt5Controller
        self.defaultSymbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']

    # upload data
    def uploadDatas(self, symbols, startTime=(2022, 8, 31, 0, 0), endTime=(2022, 10, 27, 0, 0)):
        """
        :param symbols: list for the symbols
        :param startTime: tuple for time
        :param endTime: tuple for time
        :return:
        """
        # getting data from MT5
        Prices = self.mT5Controller.pricesLoader.getPrices(symbols=symbols, start=startTime, end=endTime, timeframe='1min', count=0, ohlcvs='111111')
        # process into dataframe
        dfs = Prices.getOhlcvsFromPrices(symbols)
        # get to upload the data to database
        for symbol, df in dfs.items():
            self.postForexData(df, tableName=symbol.lower() + '_1m')

    # get data
    def downloadData(self, symbol, startTime=(2022, 8, 31, 0, 0), endTime=(2022, 10, 27, 0, 0), timeframe='5min'):
        """
        :param symbol: str
        :param startTime: tuple for time
        :param endTime: tuple for time
        :param timeframe: str
        :return:
        """
        forexDataDf = pd.DataFrame()
        # define the table name
        tableName = symbol.lower() + '_1m'
        forexDataDf_raw = self.getForexData(tableName=tableName, dateFrom=startTime, dateTo=endTime)
        for col in forexDataDf_raw:
            if col == 'open':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).first()
            elif col == 'high':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).max()
            elif col == 'low':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).min()
            elif col == 'close':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'volume':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).sum()
            elif col == 'spread':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'base_exchg':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'quote_exchg':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'ptDv':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).sum()
        # drop the nan rows that is holiday
        return forexDataDf.dropna()


# mt5Controller = MT5Controller()
# dataFeeder = DataFeeder(mt5Controller)
#
# dataFeeder.downloadData('AUDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
