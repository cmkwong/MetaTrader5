import RL.envs.TechnicalForexEnv
import config
from datetime import datetime

import os

from mt5f.MT5Controller import MT5Controller

now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
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
    'hist_bins': 100,
    'local_min_path': os.path.join(options['docs_path'], "min_data"),
    'local': False,
}

RL_options = {
    'load_net': False,
    'lr': 0.001,
    'dt_str': '220515093044',  # time that program being run
    'net_file': 'checkpoint-2970000.loader',
    'batch_size': 1024,
    'epsilon_start': 1.0,
    'epsilon_end': 0.35,
    'gamma': 0.9,
    'reward_steps': 2,
    'net_saved_path': os.path.join(options['docs_path'], "net"),
    'val_save_path': os.path.join(options['docs_path'], "val"),
    'runs_save_path': os.path.join(options['docs_path'], "runs"),
    'buffer_save_path': os.path.join(options['docs_path'], "buffer"),
    'replay_size': 100000,
    'monitor_buffer_size': 10000,
    'replay_start': 10000,  # 10000
    'epsilon_step': 1000000,
    'target_net_sync': 1000,
    'validation_step': 50000,
    'checkpoint_step': 30000,
    'weight_visualize_step': 1000,
    'buffer_monitor_step': 100000,
    'validation_episodes': 5,
}

tech_params = {
    'ma': [5, 10, 25, 50, 100, 150, 200, 250],
    'bb': [(20, 2, 2, 0), (20, 3, 3, 0), (20, 4, 4, 0), (40, 2, 2, 0), (40, 3, 3, 0), (40, 4, 4, 0)],
    'std': [(5, 1), (20, 1), (50, 1), (100, 1), (150, 1), (250, 1)],
    'rsi': [5, 15, 25, 50, 100, 150, 250],
    'stocOsci': [(5, 3, 3, 0, 0), (14, 3, 3, 0, 0), (21, 14, 14, 0, 0)],
    'macd': [(12, 26, 9), (19, 39, 9)]
}

mt5Controller = MT5Controller(timezone=data_options['timezone'], deposit_currency=data_options['deposit_currency'])
# get the loader
Prices = mt5Controller.pricesLoader.getPrices(symbols=data_options['symbols'],
                                              start=data_options['start'],
                                              end=data_options['end'],
                                              timeframe=data_options['timeframe']
                                              )

# split into train set and test set
Train_Prices, Test_Prices = mt5Controller.pricesLoader.split_Prices(Prices, percentage=data_options['trainTestSplit'])

# build the env (long)
env = RL.envs.TechnicalForexEnv.TechnicalForexEnv(data_options['symbols'][0], Train_Prices, tech_params, True,
                                                  mt5Controller.pricesLoader.all_symbol_info, 0.05, 8, 15, random_ofs_on_reset=True, reset_on_close=True)
env_val = RL.envs.TechnicalForexEnv.TechnicalForexEnv(data_options['symbols'][0], Test_Prices, tech_params, True,
                                                      mt5Controller.pricesLoader.all_symbol_info, 0.05, 8, 15, random_ofs_on_reset=False, reset_on_close=False)

