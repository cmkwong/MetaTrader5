from production.codes.models import mt5Model
import numpy as np
from production.codes import config
from production.codes.utils import tools

def create_indexes(batch_size, seq_len, data_total, shuffle):
    batch_indexes = np.empty((batch_size, data_total - seq_len), dtype=int)
    sequence = [i for i in range(seq_len, data_total)]  # start from seq_len
    # create batch indexes
    for b in range(batch_size):
        rotated_sequence = tools.shift_list(sequence, b)
        batch_indexes[b, :] = np.array(rotated_sequence)
        if shuffle: # shuffle the index if needed
            np.random.shuffle(batch_indexes[b, :])
    return batch_indexes

def get_batches(prices_matrix, seq_len, batch_size, shuffle):

    batch_indexes = create_indexes(batch_size, seq_len, len(prices_matrix), shuffle)
    # create batch
    episode_batches = []
    for i in range(len(prices_matrix) - seq_len):
        batch = np.empty((batch_size, seq_len, prices_matrix.shape[1]), dtype=float)
        indexes = batch_indexes[:,i]
        for b, index in enumerate(indexes):
            batch[b,:,:] = prices_matrix[(index-seq_len):index, :]
        episode_batches.append(batch)
    return episode_batches

data_options = {
    'start': config.START,
    'end': config.END,
    'symbols': ["EURUSD", "USDJPY"],
    'timeframe': config.TIMEFRAME,
    'timezone': config.TIMEZONE
}

prices_matrix = mt5Model.get_prices_matrix(data_options['start'], data_options['end'], data_options['symbols'],
                                           data_options['timeframe'], data_options['timezone'])
batches = get_batches(prices_matrix, seq_len=20, batch_size=32, shuffle=False)
print()

