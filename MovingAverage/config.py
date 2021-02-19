import MetaTrader5 as mt5

from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

# Moving Average
VERSION = 1
MAIN_PATH = "C://Users//user//projects//MetaTrader5//production//docs"

SYMBOL = "USDJPY"
START = (2020,1,1,0,0)
END = (2020,12,30,0,0)
TIMEFRAME = mt5.TIMEFRAME_M30
LONG_MODE = True

START_STRING = str(START[0]) + str(START[1]).zfill(2) + str(START[2]).zfill(2) + str(START[3]).zfill(2) + str(START[4]).zfill(2)
END_STRING = str(END[0]) + str(END[1]).zfill(2) + str(END[2]).zfill(2) + str(END[3]).zfill(2) + str(END[4]).zfill(2)
CSV_FILE_PATH = MAIN_PATH + "//" + str(VERSION) + "//"
CSV_FILE_NAME = "T{}_Frame-{}_Long-{}_From-{}-to-{}_result.csv".format(DT_STRING, str(TIMEFRAME), str(LONG_MODE), START_STRING, END_STRING)