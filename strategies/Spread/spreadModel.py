from mt5f.executor import common as mt5common
import pandas as pd

def get_spreads(symbols, start, end, timezone):
    """
    :param symbols: [str]
    :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
    :param timezone: 'HongKong'
    :return: pd.DataFrame
    """
    spreads = pd.DataFrame()
    for symbol in symbols:
        tick_frame = mt5common.get_ticks_range(symbol, start, end, timezone)
        spread = mt5common.get_spread_from_ticks(tick_frame, symbol)
        spreads = pd.concat([spreads, spread], axis=1, join='outer')
    spreads.columns = symbols
    return spreads