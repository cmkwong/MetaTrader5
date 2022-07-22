from dataclasses import dataclass
import pandas as pd

@dataclass
class InitPrices:
    c: pd.DataFrame
    cc: pd.DataFrame
    ptDv: pd.DataFrame
    quote_exchg: pd.DataFrame
    o: pd.DataFrame=None
    h: pd.DataFrame=None
    l: pd.DataFrame=None
    volume: pd.DataFrame=None
    spread: pd.DataFrame=None
    base_exchg: pd.DataFrame=None