from production.codes import config
from production.codes.models import mt5Model, plotModel, coinModel, fileModel
from production.codes.models.backtestModel import priceModel
from production.codes.views import plotView
import os

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
    'debug': True,
    'local': True
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["CADJPY", "USDCAD","AUDJPY", "AUDUSD"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
    'plt_save_path': os.path.join(options['main_path'], "coin_plt"),
    'debug_path': os.path.join(options['main_path'], "debug"),
    'local_min_path': os.path.join(options['main_path'], "min_data"),
}
train_options = {
    'upper_th': 3.0,
    'lower_th': -3.0,
    'z_score_mean_window': 10,
    'z_score_std_window': 20,
    'slsp': (-5000,5000), # None means no constraint
}

with mt5Model.Helper():

    # define loader
    prices_loader = priceModel.Prices_Loader(symbols=data_options['symbols'],
                                             timeframe=data_options['timeframe'],
                                             data_path=data_options['local_min_path'],
                                             start=data_options['start'],
                                             end=data_options['end'],
                                             timezone=data_options['timezone'],
                                             deposit_currency=data_options['deposit_currency'])
    # get the data
    prices_loader.get_data(options['local'])

    # split into train set and test set
    Train_Prices, Test_Prices = priceModel.split_Prices(prices_loader.Prices, percentage=data_options['trainTestSplit'])

    # get Linear Regression coefficients (independent variable and dependent variable)
    coefficient_vector = coinModel.get_coefficient_vector(Train_Prices.cc.values[:, :-1], Train_Prices.cc.values[:, -1])

    # fileModel.clear_files(data_options['extra_path']) # clear the files
    train_plt_datas = plotModel.get_coin_NN_plt_datas(Train_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                      train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['slsp'], data_options['timeframe'],
                                                      debug_path=data_options['debug_path'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])
    test_plt_datas = plotModel.get_coin_NN_plt_datas(Test_Prices, prices_loader.min_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                     train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['slsp'], data_options['timeframe'],
                                                     debug_path=data_options['debug_path'], debug_file='{}_test.csv'.format(options['dt']), debug=options['debug'])

    # save the plot
    title = plotModel.get_plot_title(data_options['start'], data_options['end'], data_options['timeframe'])
    setting = plotModel.get_setting_txt(train_options)
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0, data_options['plt_save_path'],
                       options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(40, 56), fontsize=6, bins=500,
                       setting=setting, hist_range=train_options['slsp'])

print("Saved successfully. \n{}".format(data_options['plt_save_path']))