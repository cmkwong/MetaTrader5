import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
import config
from mt5f.executor import mt5Model
from backtest import plotPre
from mt5f.loader import MT5PricesLoader
from views import plotView
from datetime import datetime
import os
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
    'dt': DT_STRING,
    'debug': True,
}

data_options = {
    'start': (2010, 1, 2, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'trainTestSplit': 0.7,
    'hist_bins': 100,
    'plt_save_path': os.path.join(options['docs_path'], "ma_backtest"),
    'debug_path': os.path.join(options['docs_path'], "debug"),
    'local_min_path': os.path.join(options['docs_path'], "min_data"),
    'local': False,
}
train_options = {
    'long_mode': True,
    'limit_unit': 3,
    'long_param': {
        'fast': 3,
        'slow': 60
    },
    'short_param': {
        'fast': 3,
        'slow': 60
    }
}

with mt5Model.csv_Writer_Helper():
    # define loader
    prices_loader = MT5PricesLoader.MT5PricesLoader(symbols=data_options['symbols'],
                                                    timeframe=data_options['timeframe'],
                                                    data_path=data_options['local_min_path'],
                                                    start=data_options['start'],
                                                    end=data_options['end'],
                                                    timezone=data_options['timezone'],
                                                    deposit_currency=data_options['deposit_currency'])
    # get the loader
    prices_loader.get_data(data_options['local'])

    # split into train set and test set
    Train_Prices, Test_Prices = prices_loader.split_Prices(prices_loader.Prices, percentage=data_options['trainTestSplit'])

    train_plt_datas = plotPre.get_ma_plt_datas(Train_Prices, train_options['long_param'], train_options['short_param'], train_options['limit_unit'],
                                               debug_path=data_options['debug_path'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])
    test_plt_datas = plotPre.get_ma_plt_datas(Test_Prices, train_options['long_param'], train_options['short_param'], train_options['limit_unit'],
                                              debug_path=data_options['debug_path'], debug_file='{}_test.csv'.format(options['dt']), debug=options['debug'])

    title = plotPre.get_plot_title(data_options['start'], data_options['end'], data_options['timeframe'], data_options['local'])
    setting = plotPre.get_setting_txt(train_options)
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0, data_options['plt_save_path'],
                       options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(42, 18), fontsize=5, bins=data_options['hist_bins'],
                       setting=setting)

    print("Saved successfully. \n{}".format(data_options['plt_save_path']))