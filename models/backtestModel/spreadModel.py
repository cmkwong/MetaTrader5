from production.codes.models import mt5Model
import pandas as pd

def get_spreads(symbols, start, end, timezone):
    """
    :param symbols: [str]
    :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param end (local time): tuple (year, month, day, hour, mins), if None, then take data until present
    :param timezone: 'HongKong'
    :return: pd.DataFrame
    """
    spreads = pd.DataFrame()
    for symbol in symbols:
        tick_frame = mt5Model.get_ticks_range(symbol, start, end, timezone)
        spread = mt5Model.get_spread_from_ticks(tick_frame, symbol)
        spreads = pd.concat([spreads, spread], axis=1, join='outer')
    spreads.columns = symbols
    return spreads