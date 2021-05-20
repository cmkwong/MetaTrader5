from production.codes import config
from production.codes.controllers import mt5Controller
from production.codes.models import mt5Model, plotModel, coinModel
from production.codes.models.backtestModel import statModel, signalModel
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
    'shuffle': True,
    'trainTestSplit': 0.7,
}
train_options = {
    'price_plt_save_path': options['main_path'] + "coin_simple_plt/",
    'dt': DT_STRING,
    'upper_th': 0.3,
    'lower_th': -0.1,
}
with mt5Controller.Helper():

    Prices = mt5Model.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                           data_options['start'], data_options['end'], deposit_currency='USD')

    # split into train set and test set
    Train_Prices, Test_Prices = mt5Model.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    # get Linear Regression coefficients
    coefficient_vector = coinModel.get_coefficient_vector(Train_Prices.c.values[:, :-1], Train_Prices.c.values[:, -1])

    # get coin data: predict, spread, z_score
    # train_coin_data = coinModel.get_coin_data(Train_Prices.c, coefficient_vector)
    # test_coin_data = coinModel.get_coin_data(Test_Prices.c, coefficient_vector)

    # train_long_signal, train_short_signal = signalModel.get_coin_signal(train_coin_data, upper_th=0.3, lower_th=-0.1)
    # test_long_signal, test_short_signal = signalModel.get_coin_signal(test_coin_data, upper_th=0.3, lower_th=-0.1)
    #
    # train_stats = statModel.get_stats(Train_Prices, train_long_signal, train_short_signal, coefficient_vector)
    # test_stats = statModel.get_stats(Test_Prices, test_long_signal, test_short_signal, coefficient_vector)

    train_plt_data = plotModel.get_coin_plt_data(Train_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'])
    test_plt_data = plotModel.get_coin_plt_data(Test_Prices, coefficient_vector, train_options['upper_th'], train_options['lower_th'])

    # save the plot
    title = plotModel.get_coin_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
    plotView.save_plot(train_plt_data, test_plt_data, data_options['symbols'], 0, train_options['price_plt_save_path'],
                       train_options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(56,24))

    # debug checking
    # returnModel.get_ret(Train_Prices.o, Train_Prices.quote_exchg, coefficient_vector, long_mode=True)
    # all_df = plotModel.append_all_df_debug(
    #     [Train_Prices.o, Train_Prices.ptDv, Train_Prices.quote_exchg, Train_Prices.base_exchg, ret_df, train_coin_data,
    #      train_long_signal, train_int_long_signal, earning, earning_by_signal])

print("Saved successfully. \n{}".format(train_options['price_plt_save_path']))