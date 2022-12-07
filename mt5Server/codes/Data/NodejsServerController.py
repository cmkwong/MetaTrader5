import requests
import pandas as pd

from myBacktest import timeModel

# upload the forex loader
class NodejsServerController:

    def __init__(self):
        self.mainUrl = "http://localhost:3002/"
        self.uploadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/upload?tableName={}"
        self.downloadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/download?tableName={}"
        self.createTableUrl = self.mainUrl + "api/v1/query/forexTable/create?tableName={}"

    def postForexData(self, forexDf: pd.DataFrame, *, tableName: str):
        """
        upload forex Data ohlcvs: open, high, low, close, volume, spread
        """
        forexDf = forexDf.fillna('')
        forexDf['datetime'] = forexDf.index
        forexDf['datetime'] = forexDf['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        listData = forexDf.to_dict('records')
        r = requests.post(self.uploadForexDataUrl.format(tableName), json={'data': listData})
        if r.status_code != 200:
            print(r.text)
            return False
        print(f"Data being uploaded: {len(listData)}")
        return True

    def getForexData(self, *, tableName: str, dateFrom: tuple, dateTo: tuple):
        """
        download forex ohlcvs from server
        :param tableName: str
        :param dateFrom: (yyyy, mm, dd, HH, MM)
        :param dateTo: (yyyy, mm, dd, HH, MM)
        :return pd.DataFrame with ohlcvs
        """
        dateFromStr = timeModel.get_time_string(dateFrom, format='yyyy-mm-dd HH:MM:ss')
        dateToStr = timeModel.get_time_string(dateTo, format='yyyy-mm-dd HH:MM:ss')
        body = {
            'from': dateFromStr,
            'to': dateToStr
        }
        r = requests.get(self.downloadForexDataUrl.format(tableName), json=body)
        res = r.json()
        if r.status_code != 200:
            print(r.text)
            return False
        print(f"Data being downloaded: {len(res['data'])}")
        # change to dataframe
        forexDf = pd.DataFrame.from_dict(res['data']).set_index('datetime')
        # change index into datetimeIndex
        forexDf.index = pd.to_datetime(forexDf.index)
        return forexDf

    def createForexTable(self, *, tableName:str):
        """
        :param tableName:
        :param colObj:
        :return:
        """
        body = {
            "schemaObj": {
                "datetime": "DATETIME NOT NULL PRIMARY KEY",
                "open": "FLOAT",
                "high": "FLOAT",
                "low": "FLOAT",
                "close": "FLOAT",
                "volume": "FLOAT",
                "spread": "FLOAT",
                "base_exchg": "FLOAT",
                "quote_exchg": "FLOAT"
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
# from Mt5f.MT5Controller import MT5Controller
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
#                                        local=data_options['local'],
#                                        ohlcvs='111111'
#                                        )
#
# dataController = DataController()
# for key, df in mt5Controller.mt5PricesLoader.Prices.rawDfs.items():
#     # dataController.uploadForexData(tableName='eurusd_1h_220717', forexDf=df)
#     # dataController.downloadForexData(tableName='eurusd_1h_220717', dateFrom="2010-01-01 00:00:00", dateTo="2022-05-01 23:59:59")
#     dataController.createForexTable(tableName='eurusd_1h_220717_2')

# ---------------------- TEST END ---------------------------