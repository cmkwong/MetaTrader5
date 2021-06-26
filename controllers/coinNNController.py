from production.codes.models import mt5Model, batchModel, coinNNModel, plotModel, timeModel, fileModel
from production.codes.models.backtestModel import priceModel
from production.codes.views import printStat, plotView
from production.codes import config
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import os

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
    'model_type': 0, # 0=FC, 1=LSTM
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"], # last one is target
    'timeframe': timeModel.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
    'seq_len': 1,                       # need to change when switch to another model
    'batch_size': 32,
    'test_epiosdes': 5,
    'check_price_plot': 5,
    'plt_save_path': os.path.join(options['main_path'], "coin_NN_plt"),
    'extra_path': os.path.join(options['main_path'], "min_data//extra_data"),
    'tensorboard_save_path': options['main_path'] + "runs/",
}
Model_options = {
    'lr': 0.01,
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
}
# tensorboard --logdir C:\Users\Chris\projects\210215_mt5\production\docs\1\runs --host localhost

with mt5Model.Helper():

    Prices = priceModel.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'], data_options['start'], data_options['end'], deposit_currency=data_options['deposit_currency'])

    # split into train set and test set
    Train_Prices, Test_Prices = priceModel.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    myModel = None
    if options['model_type'] == 0:
        myModel = coinNNModel.FC_NN(FC_options['input_size'])
    elif options['model_type'] == 1:
        myModel = coinNNModel.LSTM(LSTM_options['input_size'], LSTM_options['hidden_size'], LSTM_options['layer'], LSTM_options['batch_first'])
    optimizer = optim.Adam(myModel.parameters(), lr=Model_options['lr'])
    writer = SummaryWriter(log_dir=data_options['tensorboard_save_path'] + options['dt'], comment="coin")
    trainer = coinNNModel.Trainer(myModel, optimizer)

    episode = 1
    while True:
        train_batch = batchModel.get_batches(Train_Prices.c.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
        train_loss = trainer.run(train_batch.input, train_batch.target, train_mode=True)
        printStat.loss_status(writer, train_loss, episode, mode='train')

        if episode % data_options['test_epiosdes'] == -1:
            test_batch = batchModel.get_batches(Test_Prices.c.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
            test_loss = trainer.run(test_batch.input, test_batch.target, train_mode=False)
            printStat.loss_status(writer, test_loss, episode, mode='test')

        if episode % data_options['check_price_plot'] == 0:

            coefficient_vector = coinNNModel.get_coefficient_vector(myModel) # coefficient_vector got from neural network

            fileModel.clear_files(data_options['extra_path'])  # clear the files
            train_plt_datas = plotModel.get_coin_NN_plt_datas(Train_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                              train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['slsp'],
                                                              extra_path=data_options['extra_path'], extra_file='{}_train.csv'.format(options['dt']))
            test_plt_datas = plotModel.get_coin_NN_plt_datas(Test_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                             train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['slsp'],
                                                             extra_path=data_options['extra_path'], extra_file='{}_test.csv'.format(options['dt']))

            title = plotModel.get_plot_title(data_options['start'], data_options['end'], timeModel.get_timeframe2txt(data_options['timeframe']))
            plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], episode,
                               data_options['plt_save_path'], options['dt'], dpi=500, linewidth=0.2,
                               title=title, figure_size=(56, 24))

        episode += 1

