import pandas as pd
import numpy as np

def get_points_dff_values_df(symbols, open_prices, all_symbols_info, temp_col_name=None):
    """
    :param symbols: [str]
    :param open_prices: pd.Dataframe with open price
    :param all_symbols_info: tuple, mt5.symbols_get(). The info including the digits.
    :param temp_col_name: set None to use the symbols as column names. Otherwise, rename as fake column name
    :return: points_dff_values_df, new pd.Dataframe
    take the difference from open price
    """
    if type(open_prices) == pd.Series: open_prices = pd.DataFrame(open_prices, index=open_prices.index) # avoid the error of "too many index" if len(symbols) = 1
    points_dff_values_df = pd.DataFrame(index=open_prices.index)
    for c, symbol in enumerate(symbols):
        digits = all_symbols_info[symbol].digits # (note 44b)
        points_dff_values_df[symbol] = (open_prices.iloc[:,c] - open_prices.iloc[:,c].shift(periods=1)) * (10 ** digits) * all_symbols_info[symbol].pt_value
    if temp_col_name != None:
        points_dff_values_df.columns = [temp_col_name] * len(symbols)
    return points_dff_values_df

def get_points_dff_from_values(symbols, news, olds, all_symbols_info):
    pt_values = np.zeros((len(symbols),))
    for i, (symbol, new, old) in enumerate(zip(symbols, news, olds)):
        digits = all_symbols_info[symbol].digits
        pt_values[i] = (new - old) * (10 ** digits) * all_symbols_info[symbol].pt_value
    return pt_values