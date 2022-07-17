import requests
import pandas as pd

from myUtils.DfModel import DfModel
from data.EndPoints import EndPoints


# upload the forex loader
class DataController(EndPoints, DfModel):

    def uploadForexData(self, *, tableName: str, forexDf: pd.DataFrame):
        """
        upload forex data ohlc
        """
        forexDf['datetime'] = forexDf.index
        forexDf['datetime'] = forexDf['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        listData = self.df2ListDict(forexDf)
        r = requests.post(self.uploadForexDataUrl.format(tableName), json={'data': listData})
        if r.status_code != 200:
            print(r.text)
            return False
        print(f"Data being uploaded: {len(listData)}")
        return True

    def downloadForexData(self, *, tableName: str, dateFrom: str, dateTo: str):
        """
        download forex ohlc from server
        :return pd.DataFrame with ohlc
        """
        body = {
            'from': dateFrom,
            'to': dateTo
        }
        r = requests.get(self.downloadForexDataUrl.format(tableName), json=body)
        res = r.json()
        if r.status_code != 200:
            print(r.text)
            return False
        print(f"Data being downloaded: {len(res['data'])}")
        forexDf = pd.DataFrame.from_dict(res['data']).set_index('datetime')
        return forexDf

    def createForexTable(self, *, tableName:str):
        """
        :param tableName:
        :param colObj:
        :return:
        """
        body = {
            "colObj": {
                "datetime": "DATETIME NOT NULL PRIMARY KEY",
                "open": "FLOAT",
                "high": "FLOAT",
                "low": "FLOAT",
                "close": "FLOAT"
            }
        }
        r = requests.get(self.createTableUrl.format(tableName), json=body)
        if r.status_code != 200:
            print(r.text)
            return False
        res = r.json()
        print(res)
        return True

# ---------------------- TEST START ---------------------------
# import config
# from datetime import datetime
# import os
#
# from mt5f.MT5Controller import MT5Controller
#
# now = datetime.now()
# DT_STRING = now.strftime("%y%m%d%H%M%S")
#
# options = {
#     'docs_path': os.path.join(config.PROJECT_PATH, 'docs/{}/'.format(config.VERSION)),
#     'dt': DT_STRING,
#     'debug': True,
# }
#
# data_options = {
#     'start': (2010, 1, 2, 0, 0),
#     'end': (2020, 12, 30, 0, 0),
#     'symbols': ["EURUSD"],
#     'timeframe': '1H',
#     'timezone': "Hongkong",
#     'deposit_currency': 'USD',
#     'trainTestSplit': 0.7,
#     'hist_bins': 100,
#     'local_min_path': os.path.join(options['docs_path'], "min_data"),
#     'local': False,
# }
#
# RL_options = {
#     'load_net': False,
#     'lr': 0.001,
#     'dt_str': '220515093044',  # time that program being run
#     'net_file': 'checkpoint-2970000.loader',
#     'batch_size': 1024,
#     'epsilon_start': 1.0,
#     'epsilon_end': 0.35,
#     'gamma': 0.9,
#     'reward_steps': 2,
#     'net_saved_path': os.path.join(options['docs_path'], "net"),
#     'val_save_path': os.path.join(options['docs_path'], "val"),
#     'runs_save_path': os.path.join(options['docs_path'], "runs"),
#     'buffer_save_path': os.path.join(options['docs_path'], "buffer"),
#     'replay_size': 100000,
#     'monitor_buffer_size': 10000,
#     'replay_start': 10000,  # 10000
#     'epsilon_step': 1000000,
#     'target_net_sync': 1000,
#     'validation_step': 50000,
#     'checkpoint_step': 30000,
#     'weight_visualize_step': 1000,
#     'buffer_monitor_step': 100000,
#     'validation_episodes': 5,
# }
#
# tech_params = {
#     'ma': [5, 10, 25, 50, 100, 150, 200, 250],
#     'bb': [(20, 2, 2, 0), (20, 3, 3, 0), (20, 4, 4, 0), (40, 2, 2, 0), (40, 3, 3, 0), (40, 4, 4, 0)],
#     'std': [(5, 1), (20, 1), (50, 1), (100, 1), (150, 1), (250, 1)],
#     'rsi': [5, 15, 25, 50, 100, 150, 250],
#     'stocOsci': [(5, 3, 3, 0, 0), (14, 3, 3, 0, 0), (21, 14, 14, 0, 0)],
#     'macd': [(12, 26, 9), (19, 39, 9)]
# }
#
# mt5Controller = MT5Controller(data_path=data_options['local_min_path'], timezone=data_options['timezone'], deposit_currency=data_options['deposit_currency'])
# # get the loader
# mt5Controller.mt5PricesLoader.get_data(symbols=data_options['symbols'],
#                                        start=data_options['start'],
#                                        end=data_options['end'],
#                                        timeframe=data_options['timeframe'],
#                                        local=data_options['local'])
#
# dataController = DataController()
# for key, df in mt5Controller.mt5PricesLoader.Prices.rawDfs.items():
#     # dataController.uploadForexData(tableName='eurusd_1h_220717', forexDf=df)
#     # dataController.downloadForexData(tableName='eurusd_1h_220717', dateFrom="2010-01-01 00:00:00", dateTo="2022-05-01 23:59:59")
#     dataController.createForexTable(tableName='eurusd_1h_220717_2')

# ---------------------- TEST END ---------------------------