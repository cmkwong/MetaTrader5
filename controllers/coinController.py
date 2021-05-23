from production.codes import config
from production.codes.controllers import mt5Controller
from production.codes.models import mt5Model, plotModel, coinModel
from production.codes.views import plotView

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION)
}
data_options = {
    'start': (2015,1,1,0,0),
    'end': (2021,5,5,0,0),    # None = get the most current price
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'shuffle': True,
    'trainTestSplit': 0.7,
}
train_options = {
    'price_plt_save_path': options['main_path'] + "coin_plt/",
    'dt': DT_STRING,
    'upper_th': 0.5,
    'lower_th': -0.5,
    'z_score_mean_window': 10,
    'z_score_std_window': 10
}
with mt5Controller.Helper():

    Prices = mt5Model.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                           data_options['start'], data_options['end'], '1111', data_options['deposit_currency'])

    # split into train set and test set
    Train_Prices, Test_Prices = mt5Model.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    # get Linear Regression coefficients
    coefficient_vector = coinModel.get_coefficient_vector(Train_Prices.c.values[:, :-1], Train_Prices.c.values[:, -1])

    train_plt_datas = plotModel.get_coin_NN_plt_datas(Train_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'], train_options['z_score_mean_window'], train_options['z_score_std_window'])
    test_plt_datas = plotModel.get_coin_NN_plt_datas(Test_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'], train_options['z_score_mean_window'], train_options['z_score_std_window'])

    # save the plot
    title = plotModel.get_coin_NN_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0,
                       train_options['price_plt_save_path'], train_options['dt'], dpi=500, linewidth=0.2, title=title,
                       figure_size=(56, 24), fontsize=6, bins=500)

print("Saved successfully. \n{}".format(train_options['price_plt_save_path']))