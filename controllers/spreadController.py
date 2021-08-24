from production.codes.models import mt5Model, plotModel
from production.codes.views import plotView
from production.codes.models.backtestModel import spreadModel
from production.codes import config
from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
}

data_options = {
    'start': (2021,4,15,0,0),
    'end': (2021,5,5,23,59),    # None = get the most current price
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': 'tick',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'plt_save_path': options['main_path'] + "spread_plt/",
}

with mt5Model.Helper():
    spreads = spreadModel.get_spreads(data_options['symbols'], data_options['start'], data_options['end'], data_options['timezone'])
    plt_datas = plotModel.get_spread_plt_datas(spreads)

    title = plotModel.get_plot_title(data_options['start'], data_options['end'], 'tick', local=False)
    plotView.save_plot(plt_datas, None, data_options['symbols'], 0, data_options['plt_save_path'], # test_plt_data = None; note 56e
                       options['dt'], dpi=500, linewidth=0.2, title=title, figure_size=(30, 60), fontsize=20, bins=500)

print("Saved successfully. \n{}".format(data_options['plt_save_path']))
