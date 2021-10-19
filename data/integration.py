import pandas as pd

def append_dict_df(dict_df, mother_df, join='outer', filled=0):
    """
    :param mother_df: pd.DataFrame
    :param join: 'inner', 'outer'
    :param dict_df: {key: pd.DataFrame}
    :return: pd.DataFrame after concat
    """
    if not isinstance(mother_df, pd.DataFrame):
        mother_df = pd.DataFrame()
    for df in dict_df.values(): # do not care the key name: tech name
        if mother_df.empty:
            mother_df = df.copy()
        else:
            mother_df = pd.concat([mother_df, df], axis=1, join=join)
    return mother_df.fillna(filled)