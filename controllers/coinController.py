from production.codes.models import mt5Model
from production.codes import config
from production.codes.models.coinModel import batchModel, nnModel, trainerModel, monitorModel
from production.codes.utils import tools
from torch import optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_options = {
    'start': config.START,
    'end': config.END,
    'symbols': ["EURUSD", "EURSGD"],
    'timeframe': config.TIMEFRAME,
    'timezone': config.TIMEZONE,
    'shuffle': True,
    'trainTestSplit': 0.7,
    'seq_len': 20,
    'batch_size': 32,
}
model_options = {
    'input_size': len(data_options['symbols']) - 1,
    'lr': 0.01,
    'hidden_size': 64,
    'layer': 2,
    'batch_first': True,
}
train_options = {
    'test_epiosdes': 5,
    'check_plot': 5,
    'plt_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/coin_plt/".format(config.VERSION)
}

prices_matrix = mt5Model.get_prices_matrix(data_options['start'], data_options['end'], data_options['symbols'],
                                           data_options['timeframe'], data_options['timezone'])
# split into train set and test set
train_prices_matrix, test_prices_matrix = tools.split_matrix(prices_matrix, percentage=data_options['trainTestSplit'], axis=0)

lstm = nnModel.LSTM(model_options['input_size'], model_options['hidden_size'], model_options['layer'], model_options['batch_first']).double()
optimizer = optim.Adam(lstm.parameters(), lr=model_options['lr'])

trainer = trainerModel.Trainer(lstm, optimizer)

episode = 1
while True:
    train_batch = batchModel.get_batches(train_prices_matrix, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
    train_loss = trainer.run(train_batch.input, train_batch.target, data_options['batch_size'], train_mode=True)
    print("{}. Train loss: {:.6f}".format(episode, train_loss))

    if episode % train_options['test_epiosdes'] == 0:
        test_batch = batchModel.get_batches(test_prices_matrix, seq_len=data_options['seq_len'],batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
        test_loss = trainer.run(test_batch.input, test_batch.target, data_options['batch_size'], train_mode=False)
        print("{}. Test loss: {:.6f}".format(episode, test_loss))

    if episode % train_options['check_plot'] == 0:

        # data prepare
        train_plt, test_plt = {}, {}
        train_plt['inputs'] = train_prices_matrix[:,:-1]
        train_plt['predict'] = monitorModel.get_predicted_arr(train_plt['inputs'], lstm, data_options['seq_len'])
        train_plt['target'] = train_prices_matrix[:,-1]

        test_plt['inputs'] = test_prices_matrix[:, :-1]
        test_plt['predict'] = monitorModel.get_predicted_arr(test_plt['inputs'], lstm, data_options['seq_len'])
        test_plt['target'] = test_prices_matrix[:,-1]

        # combine into df
        df_plt = pd.DataFrame(columns=data_options['symbols'])
        for symbol in data_options['symbols'][:-1]:
            df_plt[symbol] = np.concatenate((train_plt['inputs'], test_plt['inputs']), axis=0).reshape(-1,)
        df_plt[data_options['symbols'][-1]] = np.concatenate((train_plt['target'], test_plt['target']), axis=0)
        df_plt['predict'] = np.concatenate((train_plt['predict'], test_plt['predict']), axis=0).reshape(-1,)
        # calculate the spread
        df_plt['spread'] = df_plt[data_options['symbols'][-1]] - df_plt['predict']

        # plot graph - prices
        linewidth = 0.2
        ax = plt.gca()
        for symbol in data_options['symbols'][:-1]:
            df_plt[symbol].plot(kind='line', style="k--", linewidth=linewidth)
        df_plt[data_options['symbols'][-1]].plot(kind='line', color='blue', linewidth=linewidth)
        df_plt['predict'].plot(kind='line', color='red', linewidth=linewidth)
        df_plt['spread'].plot(kind='line', color='darkorange', linewidth=linewidth) # plot spread
        plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                         # plot 0 reference line
        plt.savefig(train_options['plt_path'] + '{}-episodes-prices.jpg'.format(episode), dpi=500)
        plt.clf()

    episode += 1

