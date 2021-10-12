import pandas as pd

def append_dict_df(dict_df, output, join='outer', filled=0):
    """
    :param output: pd.DataFrame
    :param join: 'inner', 'outer'
    :param dict_df: {key: pd.DataFrame}
    :return: pd.DataFrame after concat
    """
    if not isinstance(output, pd.DataFrame):
        output = pd.DataFrame()
    for df in dict_df.values(): # do not care the key name: tech name
        output = pd.concat([output, df], axis=1, join=join)
    return output.fillna(filled)