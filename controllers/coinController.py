from production.codes import config
from production.codes.models import mt5Model, monitorModel
from production.codes.models.coinModel import linearRegressionModel
from production.codes.views import monitorView
from production.codes.utils import tools

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "C:/Users/Chris/projects/210215_mt5/production/docs/{}/".format(config.VERSION)
}
data_options = {
    'start': (2010,1,1,0,0),
    'end': (2021,5,4,0,0),
    'symbols': ["AUDJPY", "AUDUSD", "CADJPY", "USDCAD"],
    'timeframe': mt5Model.get_txt2timeframe('H4'),
    'timezone': "Etc/UTC",
    'shuffle': True,
    'trainTestSplit': 0.7,
}
train_options = {
    'price_plt_save_path': options['main_path'] + "coin_simple_plt/",
    'dt': DT_STRING
}
title = monitorView.get_plot_title(data_options['start'], data_options['end'], mt5Model.get_timeframe2txt(data_options['timeframe']))

prices_matrix = mt5Model.get_prices_matrix(data_options['start'], data_options['end'], data_options['symbols'],
                                           data_options['timeframe'], data_options['timezone'])
# split into train set and test set
train_prices_matrix, test_prices_matrix = tools.split_matrix(prices_matrix, percentage=data_options['trainTestSplit'], axis=0)

# get solution
x = linearRegressionModel.get_x_vector(train_prices_matrix[:,:-1], train_prices_matrix[:,-1])

# get predicted value
train_plt_data = monitorModel.get_plotting_data_simple(train_prices_matrix, x)
test_plt_data = monitorModel.get_plotting_data_simple(test_prices_matrix, x)
df_plt = monitorModel.get_plotting_df(train_plt_data, test_plt_data, data_options['symbols'])
monitorView.save_plot(df_plt, data_options['symbols'], 0, train_options['price_plt_save_path'], train_options['dt'],
                      len(train_plt_data['inputs']), dpi=500, linewidth=0.2, title=title, show_inputs=False)
print()