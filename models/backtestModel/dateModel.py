from production.codes.models.backtestModel import indexModel


def get_action_date(df, signal):
    """
    :param signal: Series(Boolean) without time index
    :return: start_date_list, end_date_list
    """
    start_date_list, end_date_list = [], []
    # int_signal = signal.astype(int).diff(1)
    start_index, end_index = indexModel.get_start_end_index(signal)
    # buy date
    dates = list(df['time'][start_index])
    start_date_list.extend([str(date) for date in dates])

    # sell date
    dates = list(df['time'][end_index])
    end_date_list.extend([str(date) for date in dates])

    return start_date_list, end_date_list