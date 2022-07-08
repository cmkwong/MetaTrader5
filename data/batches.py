import numpy as np
import collections
from mt5utils import tools

def create_indexes(batch_size, seq_len, data_total, shuffle=True):
    """
    Note 68b
    :param batch_size: int
    :param seq_len: int
    :param data_total: int
    :param shuffle: Boolean
    :return: np.array, indexes
    """
    batch_indexes = np.empty((batch_size, data_total - seq_len), dtype=int)
    sequence = [i for i in range(seq_len, data_total)]  # start from seq_len
    # create batch indexes
    for b in range(batch_size):
        rotated_sequence = tools.shift_list(sequence, b)
        batch_indexes[b, :] = np.array(rotated_sequence)
        if shuffle: # shuffle the index if needed
            np.random.shuffle(batch_indexes[b, :])
    return batch_indexes

def get_target_batches(arr, batch_indexes):
    target_size = batch_indexes.shape[1] # whole data length
    target_batches = []
    for t in range(target_size):
        batch_index = batch_indexes[:, t]
        target_batches.append(arr[batch_index])

    return target_batches

def get_input_batches(arr, seq_len, batch_size, batch_indexes):
    input_batches = []
    for i in range(len(arr) - seq_len):
        batch = np.empty((batch_size, seq_len, arr.shape[1]), dtype=float)
        indexes = batch_indexes[:, i]
        for b, index in enumerate(indexes):
            batch[b, :, :] = arr[(index - seq_len):index, :]
        input_batches.append(batch)
    return input_batches

def get_batches(prices_matrix, seq_len, batch_size, shuffle):
    """
    :param prices_matrix: np.array (last column must be target column)
    :param seq_len: int
    :param batch_size: int
    :return: collection batches
    """
    batches = collections.namedtuple("batches", ["input", "target"])
    input_matrix, target_matrix = prices_matrix[:, :-1], prices_matrix[:, -1] # split the data along column, last column is target
    batch_indexes = create_indexes(batch_size, seq_len, len(input_matrix), shuffle=shuffle)

    # input batch
    batches.input = get_input_batches(input_matrix, seq_len, batch_size, batch_indexes)
    # target batch
    batches.target = get_target_batches(target_matrix, batch_indexes)
    return batches