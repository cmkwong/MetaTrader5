import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
from strategies.AI.CointegrationNN import coinNNModel
from executor import mt5Model
from backtest import plotPre
from data import batches, files, prices
from views import printStat, plotView
import config
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import os

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
    'dt': DT_STRING,
    'debug': True,
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"], # last one is target
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
    'seq_len': 1,                       # need to change when switch to another model
    'batch_size': 32,
    'test_epiosdes': 5,
    'check_price_plot': 5,
    'plt_save_path': os.path.join(options['docs_path'], "coin_NN_plt"),
    'debug_path': os.path.join(options['docs_path'], "debug"),
    'tensorboard_save_path': options['docs_path'] + "runs/",
    'local': False,
}
Model_options = {
    'lr': 0.01,
    'model_type': 0, # 0=FC, 1=LSTM
}
FC_options = {
    'input_size': len(data_options['symbols']) - 1
}
LSTM_options = {
    'input_size': len(data_options['symbols']) - 1,
    'hidden_size': 64,
    'layer': 2,
    'batch_first': True,
}
train_options = {
    'upper_th': 0.3,
    'lower_th': -0.3,
    'z_score_mean_window': 3,
    'z_score_std_window': 6,
    'slsp': (-100,2000), # None means no constraint
    'close_change': 1,  # 0 = close; 1 = change
}
# tensorboard --logdir C:\Users\Chris\projects\210215_mt5\production\docs\1\runs --host localhost

with mt5Model.csv_Writer_Helper():
    prices_loader = prices.Prices_Loader(symbols=data_options['symbols'],
                                         timeframe=data_options['timeframe'],
                                         start=data_options['start'],
                                         end=data_options['end'],
                                         timezone=data_options['timezone'],
                                         data_path=data_options['local_min_path'],
                                         deposit_currency=data_options['deposit_currency'])
    # get the data
    prices_loader.get_data(data_options['local'])
    # Prices = prices_loader.get_Prices_format(options['local'])

    # split into train set and test set
    Train_Prices, Test_Prices = prices.split_Prices(prices_loader.Prices, percentage=data_options['trainTestSplit'])
    dependent_variable = Train_Prices.c
    if train_options['close_change'] == 1:
        dependent_variable = Train_Prices.cc

    # define the model, optimizer, trainer, writer
    myModel = None
    if Model_options['model_type'] == 0:
        myModel = coinNNModel.FC_NN(FC_options['input_size'])
    elif Model_options['model_type'] == 1:
        myModel = coinNNModel.LSTM(LSTM_options['input_size'], LSTM_options['hidden_size'], LSTM_options['layer'], LSTM_options['batch_first'])
    optimizer = optim.Adam(myModel.parameters(), lr=Model_options['lr'])
    writer = SummaryWriter(log_dir=data_options['tensorboard_save_path'] + options['dt'], comment="coin")
    trainer = coinNNModel.Trainer(myModel, optimizer)

    episode = 1
    while True:
        train_batch = batches.get_batches(dependent_variable.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
        train_loss = trainer.run(train_batch.input, train_batch.target, train_mode=True)
        printStat.loss_status(writer, train_loss, episode, mode='train')

        if episode % data_options['test_epiosdes'] == -1:
            test_batch = batches.get_batches(Test_Prices.c.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
            test_loss = trainer.run(test_batch.input, test_batch.target, train_mode=False)
            printStat.loss_status(writer, test_loss, episode, mode='test')

        if episode % data_options['check_price_plot'] == 0:

            coefficient_vector = coinNNModel.get_coefficient_vector(myModel) # coefficient_vector got from neural network

            files.clear_files(data_options['extra_path'])  # clear the files
            train_plt_datas = plotPre.get_coin_NN_plt_datas(Train_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                            train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['close_change'],
                                                            train_options['slsp'], data_options['timeframe'],
                                                            debug_path=data_options['debug_path'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])
            test_plt_datas = plotPre.get_coin_NN_plt_datas(Test_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                           train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['close_change'],
                                                           train_options['slsp'], data_options['timeframe'],
                                                           debug_path=data_options['debug_path'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])

            title = plotPre.get_plot_title(data_options['start'], data_options['end'], data_options['timeframe'], data_options['local'])
            plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], episode,
                               data_options['plt_save_path'], options['dt'], dpi=500, linewidth=0.2,
                               title=title, figure_size=(56, 24))

        episode += 1

