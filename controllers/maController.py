from production.codes import config
from production.codes.models import mt5Model, plotModel, priceModel
from production.codes.views import plotView
from production.codes.controllers import mt5Controller
from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION)
}

data_options = {
    'start': (2010, 1, 1, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD"],
    'timeframe': mt5Model.get_txt2timeframe('D1'),
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'trainTestSplit': 0.7,
}
train_options = {
    'price_plt_save_path': options['main_path'] + "ma_backtest/",
    'dt': DT_STRING,
    'long_mode': True,
    'limit_unit': 5,
    'bins': 100
}
long_param = {
    'fast': 2,
    'slow': 7
}
short_param = {
    'fast': 56,
    'slow': 10
}

with mt5Controller.Helper():
    Prices = priceModel.get_Prices(data_options['symbols'], data_options['timeframe'], data_options['timezone'], data_options['start'], data_options['end'], ohlc='1111', deposit_currency='USD')

    # split into train set and test set
    Train_Prices, Test_Prices = priceModel.split_Prices(Prices, percentage=data_options['trainTestSplit'])

    train_plt_datas = plotModel.get_ma_plt_datas(Train_Prices, long_param, short_param, train_options['limit_unit'])
    test_plt_datas = plotModel.get_ma_plt_datas(Test_Prices, long_param, short_param, train_options['limit_unit'])

    title = plotModel.get_coin_NN_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))
    plotView.save_plot(train_plt_datas, test_plt_datas, data_options['symbols'], 0,
                       train_options['price_plt_save_path'], train_options['dt'], dpi=500, linewidth=0.2, title=title,
                       figure_size=(42, 18), fontsize=5, bins=100)

    print("Saved successfully. \n{}".format(train_options['price_plt_save_path']))

    # df = mt5Model.get_historical_data(data_options['symbol'], data_options['timeframe'], data_options['timezone'],
    #                                   data_options['start'], data_options['end'])
    # signal = signalModel.get_movingAverage_signal(df, fast_index, slow_index, limit_unit, data_options['long_mode'])
    #
    # # information and statistic
    # details = statModel.get_action_detail(df, signal)
    # stat = statModel.get_stat(df, signal)
    # stat["fast"], stat["slow"], stat["limit"] = fast_index, slow_index, limit_unit
    # printStat.print_dict(details)
    # printStat.print_dict(stat)

    # plot graph
    # ret_list = returnModel.get_ret_list(Train_Prices.o, signal)
    # plotView.density(ret_list, bins=data_options['bins'])