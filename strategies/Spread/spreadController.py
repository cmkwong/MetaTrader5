import sys
sys.path.append('C:/Users/Chris/projects/210215_mt5')
from mt5f.executor import mt5Model
from backtest import plotPre
from strategies.Spread import spreadModel
from views import plotView
import config
from datetime import datetime
import os
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
    'dt': DT_STRING,
}

data_options = {
    'start': (2021,11,15,0,0),
    'end': (2021,11,18,23,59),    # None = get the most current price
    'symbols': ["AUDJPY", 	"AUDUSD", 	"CADJPY", 	"EURUSD", 	"NZDUSD", 	"USDCAD"],
    'timeframe': 'tick',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'plt_save_path': os.path.join(options['docs_path'], "spread_plt")
}

with mt5Model.csv_Writer_Helper():
    spreads = spreadModel.get_spreads(data_options['symbols'], data_options['start'], data_options['end'], data_options['timezone'])
    plt_datas = plotPre.get_spread_plt_datas(spreads)

    title = plotPre.get_plot_title(data_options['start'], data_options['end'], 'tick', local=False)
    plotView.save_plot(plt_datas, None, data_options['symbols'], 0, data_options['plt_save_path'], # test_plt_data = None; note 56e
                       options['dt'], dpi=250, linewidth=0.2, title=title, figure_size=(30, 60), fontsize=20, bins=500)

print("Saved successfully. \n{}".format(data_options['plt_save_path']))
