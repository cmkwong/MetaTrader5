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

