import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
import numpy as np
import config
from mt5.executor import mt5Model
from strategies.Cointegration import coinModel
from backtest import plotPre
from mt5.loader import MT5PricesLoader
from views import plotView
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
    'start': (2021,5,1,0,0),
    'end': (2021,10,25,0,0),    # None = get the most current price
    'symbols': ["AUDJPY","AUDUSD","CADJPY","USDCAD"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
    'hist_bins': 300,
    'dpi': 300,
    'plt_save_path': os.path.join(options['docs_path'], "coin_plt"),
    'debug_path': os.path.join(options['docs_path'], "debug"),
    'local_min_path': os.path.join(options['docs_path'], "min_data"),
    'local': False,
}
train_options = {
    'upper_th': 1.5,
    'lower_th': -1.5,
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': None, # None means no constraint
    'close_change': 1,  # 0 = close; 1 = change
}

print(data_options['debug_path'])
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

    # get Linear Regression coefficients (independent variable and dependent variable)
    dependent_variable = Train_Prices.c
    if train_options['close_change'] == 1:
        dependent_variable = Train_Prices.cc
    coefficient_vector = coinModel.get_coefficient_vector(dependent_variable.values[:, :-1], dependent_variable.values[:, -1])
    # debug: hard coding the coefficient factor
    coefficient_vector = np.array([0.0, 0.98467, -0.98578, -0.98662])

    # fileModel.clear_files(data_options['extra_path']) # clear the files
    train_plt_datas = plotPre.get_coin_NN_plt_datas(Train_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                    train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['close_change'],
                                                    train_options['slsp'], data_options['timeframe'],
                                                    debug_path=data_options['debug_path'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])
    test_plt_datas = plotPre.get_coin_NN_plt_datas(Test_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                   train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['close_change'],
                                                   train_options['slsp'], data_options['timeframe'],
                                                   debug_path=data_options['debug_path'], debug_file='{}_test.csv'.format(options['dt']), debug=options['debug'])

    # save the plot
    title = plotPre.get_plot_title(data_options['start'], data_options['end'], data_options['timeframe'], data_options['local'])
    setting = plotPre.get_setting_txt(train_options)
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0, data_options['plt_save_path'],
                       options['dt'], dpi=data_options['dpi'], linewidth=0.2, title=title, figure_size=(40, 56), fontsize=6, bins=data_options['hist_bins'],
                       setting=setting, hist_range=train_options['slsp'])

print("Saved successfully. \n{}".format(data_options['plt_save_path']))