from dataclasses import dataclass
import pandas as pd


@dataclass
class InitPrices:
    close: pd.DataFrame
    cc: pd.DataFrame
    ptDv: pd.DataFrame
    quote_exchg: pd.DataFrame
    base_exchg: pd.DataFrame = pd.DataFrame()
    open: pd.DataFrame = pd.DataFrame()
    high: pd.DataFrame = pd.DataFrame()
    low: pd.DataFrame = pd.DataFrame()
    volume: pd.DataFrame = pd.DataFrame()
    spread: pd.DataFrame = pd.DataFrame()

    def getValidCols(self):
        validCol = []
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if not value.empty:
                validCol.append(value)
        return validCol

    def getOhlcvsFromPrices(self, symbols=None):
        """
        resume into normal dataframe
        :param symbols: [symbol str]
        :param Prices: Prices collection
        :return: {pd.DataFrame}
        """
        # get the default symbol list
        if not symbols:
            symbols = []
            for symbol in self.close:
                symbols.append(symbol)

        ohlcsvs = {}
        nameDict = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'spread': 'spread', 'ptDv': 'ptDv', 'quote_exchg': 'quote_exchg', 'base_exchg': 'base_exchg'}
        for si, symbol in enumerate(symbols):
            requiredDf = pd.DataFrame() # create empty df
            for name, field in self.__dataclass_fields__.items():  # name = variable name; field = pd.dataframe/ value
                if name not in nameDict.keys(): continue  # only need the cols in nameDict
                df = getattr(self, name)
                if not df.empty:
                    dfCol = df.iloc[:, si].rename(nameDict[name])  # get required column
                    if requiredDf.empty:
                        requiredDf = dfCol.copy()
                    else:
                        requiredDf = pd.concat([requiredDf, dfCol], axis=1)
            ohlcsvs[symbol] = requiredDf
        return ohlcsvs
        # o = Prices.o.iloc[:, i].rename('open')
        # h = Prices.h.iloc[:, i].rename('high')
        # l = Prices.l.iloc[:, i].rename('low')
        # c = Prices.c.iloc[:, i].rename('close')
        # v = Prices.volume.iloc[:, i].rename('volume')  # volume
        # s = Prices.spread.iloc[:, i].rename('spread')  # spread
