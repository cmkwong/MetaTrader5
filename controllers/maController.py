from production.codes import config
from production.codes.models import mt5Model, plotModel, timeModel
from production.codes.models.backtestModel import priceModel
from production.codes.views import plotView
from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
}

data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD"],
    'timeframe': timeModel.get_txt2timeframe('D1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'trainTestSplit': 0.7,
    'hist_bins': 100,
    'price_plt_save_path': options['main_path'] + "ma_backtest/",
}
train_options = {
    'long_mode': True,
    'limit_unit': 5,
    'long_param': {
        'fast': 2,
        'slow': 7
    },
    'short_param': {
        'fast': 56,
        'slow': 10
    }
}

with mt5Model.Helper():
    Prices = priceModel.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'], data_options['start'], data_options['end'], ohlc='1111', deposit_currency='USD')

    # split into train set and test set
    Train_Prices, Test_Prices = priceModel.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    train_plt_datas = plotModel.get_ma_plt_datas(Train_Prices, train_options['long_param'], train_options['short_param'], train_options['limit_unit'])
    test_plt_datas = plotModel.get_ma_plt_datas(Test_Prices, train_options['long_param'], train_options['short_param'], train_options['limit_unit'])

    title = plotModel.get_plot_title(data_options['start'], data_options['end'], timeModel.get_timeframe2txt(data_options['timeframe']))
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0,
                       data_options['price_plt_save_path'], options['dt'], dpi=500, linewidth=0.2, title=title,
                       figure_size=(42, 18), fontsize=5, bins=data_options['hist_bins'])

    print("Saved successfully. \n{}".format(data_options['price_plt_save_path']))