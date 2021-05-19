from production.codes.models import mt5Model, batchModel, coinNNModel, plotModel
from production.codes.views import printStat, plotView
from production.codes.utils import tools
from production.codes import config
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION)
}
data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["AUDUSD", "EURGBP"], # last one is target
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
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
    'check_price_plot': 5,
    'price_plt_saved_path': options['main_path'] + "coin_plt/",
    'tensorboard_save_path': options['main_path'] + "runs/",
    'dt': DT_STRING
}
# tensorboard --logdir C:\Users\Chris\projects\210215_mt5\production\docs\1\runs --host localhost

prices_df = mt5Model.get_prices_df(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                       data_options['start'], data_options['end'])

# split into train set and test set
train_prices_df, test_prices_df = tools.split_df(prices_df, percentage=data_options['trainTestSplit'])

lstm = coinNNModel.LSTM(model_options['input_size'], model_options['hidden_size'], model_options['layer'], model_options['batch_first'])
optimizer = optim.Adam(lstm.parameters(), lr=model_options['lr'])
writer = SummaryWriter(log_dir=train_options['tensorboard_save_path']+train_options['dt'], comment="coin")
trainer = coinNNModel.Trainer(lstm, optimizer)

episode = 1
while True:
    train_batch = batchModel.get_batches(train_prices_df.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
    train_loss = trainer.run(train_batch.input, train_batch.target, data_options['batch_size'], train_mode=True)
    printStat.loss_status(writer, train_loss, episode, mode='train')

    if episode % train_options['test_epiosdes'] == 0:
        test_batch = batchModel.get_batches(train_prices_df.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
        test_loss = trainer.run(test_batch.input, test_batch.target, data_options['batch_size'], train_mode=False)
        printStat.loss_status(writer, test_loss, episode, mode='test')

    if episode % train_options['check_price_plot'] == 0:
        title = plotModel.get_coin_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
        train_coinNN_data = coinNNModel.get_coinNN_data(train_prices_df, lstm, data_options['seq_len'])
        test_coinNN_data = coinNNModel.get_coinNN_data(test_prices_df, lstm, data_options['seq_len'])
        plotView.save_plot(train_coinNN_data, test_coinNN_data, data_options['symbols'], episode, train_options['price_plt_saved_path'],
                  train_options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(14,6))
    episode += 1

