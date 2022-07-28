from dataclasses import dataclass
import pandas as pd


@dataclass
class InitPrices:
    c: pd.DataFrame
    cc: pd.DataFrame
    ptDv: pd.DataFrame
    quote_exchg: pd.DataFrame
    o: pd.DataFrame = pd.DataFrame()
    h: pd.DataFrame = pd.DataFrame()
    l: pd.DataFrame = pd.DataFrame()
    volume: pd.DataFrame = pd.DataFrame()
    spread: pd.DataFrame = pd.DataFrame()
    base_exchg: pd.DataFrame = pd.DataFrame()

    def getValidCols(self):
        validCol = []
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if not value.empty:
                validCol.append(value)
        return validCol

    def getOhlcvsFromPrices(self, symbols, Prices, ohlcvs):
        """
        resume into normal dataframe
        :param symbols: [symbol str]
        :param Prices: Prices collection
        :return: {pd.DataFrame}
        """
        ohlcsvs = {}
        vaildCol = Prices.getValidCols()
        for i, symbol in enumerate(symbols):
            if ohlcvs[0] == 1: o = Prices.o.iloc[:, i].rename('open')
            h = Prices.h.iloc[:, i].rename('high')
            l = Prices.l.iloc[:, i].rename('low')
            c = Prices.c.iloc[:, i].rename('close')
            v = Prices.volume.iloc[:, i].rename('volume')  # volume
            s = Prices.spread.iloc[:, i].rename('spread')  # spread
            ohlcsvs[symbol] = pd.concat([o, h, l, c, v, s], axis=1)
        return ohlcsvs
