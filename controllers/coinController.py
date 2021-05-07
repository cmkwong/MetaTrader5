from production.codes import config
from production.codes.models import mt5Model, plotModel, coinModel
from production.codes.views import plotView

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION)
}
data_options = {
    'start': (2010,1,1,0,0),
    'end': None,    # None = get the most current price
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H1'),
    'timezone': "Hongkong",
    'shuffle': True,
    'trainTestSplit': 0.7,
}
train_options = {
    'price_plt_save_path': options['main_path'] + "coin_simple_plt/",
    'dt': DT_STRING
}
title = plotModel.get_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))

prices_df = mt5Model.get_prices_df(data_options['symbols'], data_options['timeframe'], data_options['timezone'],
                                       data_options['start'], data_options['end'])

# split into train set and test set
train_prices_df, test_prices_df = mt5Model.split_df(prices_df, percentage=data_options['trainTestSplit'])

# get solution
coefficient_vector = coinModel.get_coefficient_vector(train_prices_df.values[:, :-1], train_prices_df.values[:, -1])

# get plotted graph
train_plt_df = plotModel.get_plotting_data_simple(train_prices_df, coefficient_vector)
test_plt_df = plotModel.get_plotting_data_simple(test_prices_df, coefficient_vector)
plotView.save_plot(train_plt_df, test_plt_df, data_options['symbols'], 0, train_options['price_plt_save_path'],
                   train_options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(28,12))

# get the

print()