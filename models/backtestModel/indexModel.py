from production.codes.models.backtestModel import signalModel

def get_open_index(int_signal):
    """
    :param int_signal: pd.Series, index can be any types of index
    :return: list
    """
    start_index = []
    start_index.extend([get_required_index(int_signal, index, step=1) for index in int_signal[int_signal == 1].index])  # see note point 6 why added by 1
    # start_index.extend([index + 1 for index in int_signal[int_signal == 1].index])  # see note point 6 why added by 1
    return start_index

def get_close_index(int_signal):
    """
    :param int_signal: pd.Series, index cannot be Timestamp
    :return: list
    """
    end_index = []
    end_index.extend([get_required_index(int_signal, index, step=1) for index in int_signal[int_signal == -1].index])   # see note point 6 why added by 1
    # end_index.extend([index + 1 for index in int_signal[int_signal == -1].index]) # see note point 6 why added by 1
    return end_index

def get_signal_start_index(int_signal):
    """
    :param int_signal: pd.Series, index can be Timestamp
    :return: list
    """
    start_index = []
    start_index.extend([index for index in int_signal[int_signal == 1].index])  # see note point 6 why added by 1
    return start_index

def get_signal_end_index(int_signal):
    """
    :param int_signal: pd.Series, index can be Timestamp
    :return: list
    """
    start_index = []
    start_index.extend([index for index in int_signal[int_signal == -1].index])  # see note point 6 why added by 1
    return start_index

def get_action_start_end_index(signal):
    """
    :param signal: pd.Series
    :return: list: start_index, end_index
    """
    int_signal = signalModel.get_int_signal(signal)
    # buy index
    start_index = get_open_index(int_signal)
    # sell index
    end_index = get_close_index(int_signal)
    return start_index, end_index

def simple_limit_end_index(starts, ends, limit_unit):
    """
    modify the ends_index, eg. close the trade until specific unit
    :param starts: list [int] index
    :param ends: list [int] index
    :return: starts, ends
    """
    new_starts_index, new_ends_index = [], []
    for s, e in zip(starts, ends):
        new_starts_index.append(s)
        new_end = min(s + limit_unit, e)
        new_ends_index.append(new_end)
    return new_starts_index, new_ends_index

def get_required_index(df, curr_index, step, numeric=False):
    """
    :param df: pd.Series, pd.DataFrame
    :param curr_index: index
    :param step: int, +/-
    :return: index
    """
    if numeric:
        required_index = df.index.get_loc(curr_index) + step
    else:
        required_index = df.index[df.index.get_loc(curr_index) + step]
    return required_index