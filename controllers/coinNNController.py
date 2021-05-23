from production.codes.models import mt5Model, batchModel, coinNNModel, plotModel
from production.codes.views import printStat, plotView
from production.codes.controllers import mt5Controller
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
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"], # last one is target
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
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
    'price_plt_save_path': options['main_path'] + "coin_NN_plt/",
    'tensorboard_save_path': options['main_path'] + "runs/",
    'dt': DT_STRING,
    'upper_th': 0.3,
    'lower_th': -0.3,
    'z_score_mean_window': 3,
    'z_score_std_window': 6
}
# tensorboard --logdir C:\Users\Chris\projects\210215_mt5\production\docs\1\runs --host localhost

with mt5Controller.Helper():

    Prices = mt5Model.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                 data_options['start'], data_options['end'], '1111', data_options['deposit_currency'])

    # split into train set and test set
    Train_Prices, Test_Prices = mt5Model.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    lstm = coinNNModel.LSTM(model_options['input_size'], model_options['hidden_size'], model_options['layer'], data_options['seq_len'], model_options['batch_first'])
    optimizer = optim.Adam(lstm.parameters(), lr=model_options['lr'])
    writer = SummaryWriter(log_dir=train_options['tensorboard_save_path']+train_options['dt'], comment="coin")
    trainer = coinNNModel.Trainer(lstm, optimizer)

    episode = 1
    while True:
        train_batch = batchModel.get_batches(Train_Prices.c.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
        train_loss = trainer.run(train_batch.input, train_batch.target, data_options['batch_size'], train_mode=True)
        printStat.loss_status(writer, train_loss, episode, mode='train')

        if episode % train_options['test_epiosdes'] == -1:
            test_batch = batchModel.get_batches(Test_Prices.c.values, seq_len=data_options['seq_len'], batch_size=data_options['batch_size'], shuffle=data_options['shuffle'])
            test_loss = trainer.run(test_batch.input, test_batch.target, data_options['batch_size'], train_mode=False)
            printStat.loss_status(writer, test_loss, episode, mode='test')

        if episode % train_options['check_price_plot'] == 0:

            coefficient_vector = lstm.get_coefficient_vector() # coefficient_vector got from neural network
            train_plt_datas = plotModel.get_coin_NN_plt_datas(Train_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'], train_options['z_score_mean_window'], train_options['z_score_std_window'])
            test_plt_datas = plotModel.get_coin_NN_plt_datas(Test_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'], train_options['z_score_mean_window'], train_options['z_score_std_window'])

            title = plotModel.get_coin_NN_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
            plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], episode,
                               train_options['price_plt_save_path'], train_options['dt'], dpi=500, linewidth=0.2,
                               title=title, figure_size=(56, 24))

        episode += 1

