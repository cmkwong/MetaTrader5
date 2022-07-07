from backtest import signalModel

def find_target_index(series, target, step=1, numeric=False):
    """
    :param series: pd.Series, index can be any types of index
    :param target: int (the value need to find)
    :return: list
    """
    start_index = []
    start_index.extend([get_step_index_by_index(series, index, step, numeric) for index in series[series == target].index])  # see note point 6 why added by 1
    return start_index

def get_start_end_index(signal, step=1, numeric=False):
    """
    :param signal: pd.Series
    :return: list: start_index, end_index
    """
    int_signal = signalModel.get_int_signal(signal)
    # buy index
    start_index = find_target_index(int_signal, 1, step=step, numeric=numeric)
    # sell index
    end_index = find_target_index(int_signal, -1, step=step, numeric=numeric)
    return start_index, end_index

def get_step_index_by_index(series, curr_index, step, numeric=False):
    """
    :param series: pd.Series, pd.DataFrame
    :param curr_index: index
    :param step: int, +/-
    :return: index
    """
    if numeric:
        required_index = series.index.get_loc(curr_index) + step
    else:
        required_index = series.index[series.index.get_loc(curr_index) + step]
    return required_index

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