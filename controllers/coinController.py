from production.codes import config
from production.codes.utils import tools
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
    train_prices_df, test_prices_df = tools.split_df(Prices.c, percentage=data_options['trainTestSplit'])

    # get solution
    coefficient_vector = coinModel.get_coefficient_vector(train_prices_df.values[:, :-1], train_prices_df.values[:, -1])

    # get plotted graph data
    train_plt_df = plotModel.get_plotting_data_simple(train_prices_df, coefficient_vector)
    test_plt_df = plotModel.get_plotting_data_simple(test_prices_df, coefficient_vector)

    points_dff_values_df = mt5Model.get_points_dff_values_df(Prices.o, all_symbols_info)

    coin_signal = mt5Model.get_coin_signal(train_plt_df, upper_th=0.6, lower_th=-0.2)
    int_signal = mt5Model.get_int_signal(coin_signal)

    changes = mt5Model.get_changes(Prices.ref, points_dff_values_df, coefficient_vector)

    changes_by_signal = mt5Model.get_changes_by_signal(changes, coin_signal)

    all_df = mt5Model.append_all_debug([Prices.o, points_dff_values_df, train_plt_df, coin_signal, int_signal, changes, changes_by_signal])

    # save the plot
    plotView.save_plot(train_plt_df, test_plt_df, data_options['symbols'], 0, train_options['price_plt_save_path'],
                       train_options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(28,12))

print()