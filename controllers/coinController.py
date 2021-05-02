from production.codes.models import mt5Model
from production.codes import config
from production.codes.models.coinModel import batchModel, nnModel, criterionModel
from production.codes.utils import tools
from torch import optim
import torch

options = {
    # data
    'start': config.START,
    'end': config.END,
    'symbols': ["EURUSD", "USDJPY"],
    'timeframe': config.TIMEFRAME,
    'timezone': config.TIMEZONE,
    'trainTestSplit': 0.7,
    'seq_len': 20,
    #model
    'batch_size': 32,
    'lr': 0.01,
    'hidden_size': 64,
    'layer': 2,
    'batch_first': True
}
INPUT_SIZE = len(options['symbols']) - 1
prices_matrix = mt5Model.get_prices_matrix(options['start'], options['end'], options['symbols'],
                                           options['timeframe'], options['timezone'])

# split into train set and test set
train_prices_matrix, test_prices_matrix = tools.split_matrix(prices_matrix, percentage=0.7, axis=0)
train_batch = batchModel.get_batches(train_prices_matrix, seq_len=options['seq_len'], batch_size=options['batch_size'])
test_batch = batchModel.get_batches(test_prices_matrix, seq_len=options['seq_len'], batch_size=options['batch_size'])

lstm = nnModel.LSTM(INPUT_SIZE, options['hidden_size'], options['layer'], options['batch_first']).double()
optimizer = optim.Adam(lstm.parameters(), lr=options['lr'])

for input, target in zip(train_batch.input, train_batch.target):
    lstm.train()
    input, target = torch.from_numpy(input).requires_grad_().double(), torch.from_numpy(target).requires_grad_().double()
    hiddens = lstm.init_hiddens(options['batch_size'])
    output = lstm(input, hiddens)
    loss = criterionModel.get_mse_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(loss)

print()
