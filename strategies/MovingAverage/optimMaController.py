from production.codes import config
from production.codes.executor import mt5Model
from production.codes.backtest import timeModel
from production.codes.strategies.MovingAverage import maModel
from production.codes.data import prices
from datetime import datetime
import os
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
    'debug': True,
}

data_options = {
    'start': (2010, 1, 2, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'trainTestSplit': 0.7,
    'max_index': 201,
    'max_limit_range': (0, 21),
    'append_checkpoint': 5,
    'csv_save_path': os.path.join(options['main_path'], "ma_backtest"),
    'local_min_path': os.path.join(options['main_path'], "min_data"),
    'local': False,
}

# prepare the date string and file name
curr_time_string = timeModel.get_current_time_string()
start_string = timeModel.get_time_string(data_options['start'])
end_string = timeModel.get_time_string(data_options['end'])
files_name = []
train_long_stat_file_name = "{}_{}_{}_Long_Limit{}_From{}_To{}_Train.csv".format(
    curr_time_string,
    data_options['symbols'][0],
    data_options['timeframe'],
    data_options['max_limit_range'],
    start_string,
    end_string
)
train_short_stat_file_name = "{}_{}_{}_Short_Limit{}_From{}_To{}_Train.csv".format(
    curr_time_string,
    data_options['symbols'][0],
    data_options['timeframe'],
    data_options['max_limit_range'],
    start_string,
    end_string
)
test_long_stat_file_name = "{}_{}_{}_Long_Limit{}_From{}_To{}_Test.csv".format(
    curr_time_string,
    data_options['symbols'][0],
    data_options['timeframe'],
    data_options['max_limit_range'],
    start_string,
    end_string
)
test_short_stat_file_name = "{}_{}_{}_Short_Limit{}_From{}_To{}_Test.csv".format(
    curr_time_string,
    data_options['symbols'][0],
    data_options['timeframe'],
    data_options['max_limit_range'],
    start_string,
    end_string
)
with mt5Model.csv_Writer_Helper(csv_save_path=data_options['csv_save_path'],
                                csv_file_names=[train_long_stat_file_name, train_short_stat_file_name, test_long_stat_file_name, test_short_stat_file_name],
                                append_checkpoint=data_options["append_checkpoint"]) as helper:
    # define loader
    prices_loader = prices.Prices_Loader(symbols=data_options['symbols'],
                                         timeframe=data_options['timeframe'],
                                         data_path=data_options['local_min_path'],
                                         start=data_options['start'],
                                         end=data_options['end'],
                                         timezone=data_options['timezone'],
                                         deposit_currency=data_options['deposit_currency'])
    # get the data
    prices_loader.get_data(data_options['local'])

    # split into train set and test set
    Train_Prices, Test_Prices = prices.split_Prices(prices_loader.Prices, percentage=data_options['trainTestSplit'])

    for limit_unit in range(data_options['max_limit_range'][0], data_options['max_limit_range'][1]):
        # optimizing the training set
        train_long_stat_txt, train_short_stat_txt = maModel.get_optimize_moving_average_csv_text(Train_Prices, limit_unit, max_index=data_options['max_index'])
        helper.append_txt(train_long_stat_txt, train_long_stat_file_name)
        helper.append_txt(train_short_stat_txt, train_short_stat_file_name)

        # optimizing the testing set
        test_long_stat_txt, test_short_stat_txt = maModel.get_optimize_moving_average_csv_text(Test_Prices, limit_unit, max_index=data_options['max_index'])
        helper.append_txt(test_long_stat_txt, test_long_stat_file_name)
        helper.append_txt(test_short_stat_txt, test_short_stat_file_name)

