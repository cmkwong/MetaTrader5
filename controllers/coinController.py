from production.codes import config
from production.codes.controllers import mt5Controller
from production.codes.models import mt5Model, plotModel, coinModel
from production.codes.models.backtestModel import statModel, indexModel, returnModel
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
    'dt': DT_STRING
}
with mt5Controller.Helper():
    title = plotModel.get_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))

    all_symbols_info = mt5Model.get_all_symbols_info()
    Prices = mt5Model.get_Prices(data_options['symbols'], all_symbols_info, data_options['timeframe'], data_options['timezone'],
                                           data_options['start'], data_options['end'])

    # split into train set and test set
    Train_Prices, Test_Prices = mt5Model.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    # get Linear Regression coefficients
    coefficient_vector = coinModel.get_coefficient_vector(Train_Prices.c.values[:, :-1], Train_Prices.c.values[:, -1])

    # get coin data: predict, spread, z_score
    train_coin_data = plotModel.get_coin_data_lr(Train_Prices.c, coefficient_vector)
    test_coin_data = plotModel.get_coin_data_lr(Test_Prices.c, coefficient_vector)

    train_coin_signal = mt5Model.get_coin_signal(train_coin_data, upper_th=0.6, lower_th=-0.2)
    train_int_signal = mt5Model.get_int_signal(train_coin_signal)

    earning = mt5Model.get_coin_earning(Train_Prices.quote_exchg, Train_Prices.ptDv, coefficient_vector)
    earning_by_signal = mt5Model.get_coin_earning_by_signal(earning, train_coin_signal)

    testSignal = indexModel.get_open_index(train_int_signal['long'])

    rets_df = returnModel.get_rets_df_debug(Train_Prices.o)

    all_df = mt5Model.append_all_debug([Train_Prices.o, Train_Prices.ptDv, Train_Prices.quote_exchg, Train_Prices.base_exchg, rets_df, train_coin_data, train_coin_signal, train_int_signal, earning, earning_by_signal])

    stat = statModel.get_stat(Train_Prices.o, earning['long_earning'], train_coin_signal['long_signal'], coefficient_vector)
    # save the plot
    plotView.save_plot(train_plt_df, test_plt_df, data_options['symbols'], 0, train_options['price_plt_save_path'],
                       train_options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(28,12))

print()