import MetaTrader5 as mt5

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

# Moving Average
VERSION = 1
MAIN_PATH = "C://Users//user//projects//MetaTrader5//production//docs"

SYMBOL = "USDJPY"
START = (2009,10,1,0,0)
END = (2010,11,1,0,0)
TIMEFRAME = mt5.TIMEFRAME_M10
LONG_MODE = True

CSV_FILE_PATH = MAIN_PATH + "//" + str(VERSION) + "//"
CSV_FILE_NAME = "T{}_F{}_Long-{}_result.csv".format(DT_STRING, str(TIMEFRAME), str(LONG_MODE))