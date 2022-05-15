import numpy as np
import os

def shift_list(lst, s):
    s %= len(lst)
    s *= -1
    shifted_lst = lst[s:] + lst[:s]
    return shifted_lst

def get_accuracy(values, th=0.0):
    """
    :param values: listclose_price_with_last_tick
    :param th: float
    :return: float
    """
    accuracy = np.sum([c > th for c in values]) / len(values)
    return accuracy

def append_dict_into_text(stat, txt=''):
    values = list(stat.values())
    txt += ','.join([str(value) for value in values]) + '\n'
    return txt

def find_required_path(path, target):
    while(True):
        head, tail = os.path.split(path)
        if tail == target:
            return path
        path = head

