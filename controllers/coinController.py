from production.codes import config
from production.codes.controllers import mt5Controller
from production.codes.models import mt5Model, plotModel, coinModel
from production.codes.views import plotView

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION),
    'dt': DT_STRING,
    'debug': True
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
    'price_plt_save_path': options['main_path'] + "coin_plt/",
}
train_options = {
    'upper_th': 0.3,
    'lower_th': -0.3,
    'z_score_mean_window': 5,
    'z_score_std_window': 20,
    'slsp': (-100,5000), # None means no constraint
}

with mt5Controller.Helper():

    Prices = mt5Model.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                           data_options['start'], data_options['end'], '1111', data_options['deposit_currency'])

    # split into train set and test set
    Train_Prices, Test_Prices = mt5Model.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    # get Linear Regression coefficients
    coefficient_vector = coinModel.get_coefficient_vector(Train_Prices.c.values[:, :-1], Train_Prices.c.values[:, -1])

    train_plt_datas = plotModel.get_coin_NN_plt_datas(Train_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                      train_options['z_score_mean_window'], train_options['z_score_std_window'],
                                                      train_options['slsp'], debug_file='{}_train.csv'.format(options['dt']), debug=options['debug'])
    test_plt_datas = plotModel.get_coin_NN_plt_datas(Test_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'],
                                                     train_options['z_score_mean_window'], train_options['z_score_std_window'], train_options['slsp'],
                                                     debug_file='{}_test.csv'.format(options['dt']), debug=options['debug'])

    # save the plot
    title = plotModel.get_coin_NN_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
    setting = plotModel.get_setting_txt(train_options)
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0, data_options['price_plt_save_path'],
                       options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(56, 24), fontsize=6, bins=500,
                       setting=setting, hist_range=train_options['slsp'])

print("Saved successfully. \n{}".format(data_options['price_plt_save_path']))