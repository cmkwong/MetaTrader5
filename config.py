import MetaTrader5 as mt5
from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")
TIMEFRAME_DICT = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
                  "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
                  "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
                  "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
                  "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
                  "MN1": mt5.TIMEFRAME_MN1}

# Moving Average
VERSION = 1
MAIN_PATH = "/production/docs"

TIMEZONE = "Etc/UTC"
SYMBOL = "EURUSD"
START = (2010,1,1,0,0)
END = (2020,12,30,0,0)
TIMEFRAME_TEXT = "H4"
LONG_MODE = True
LIMIT_UNIT = 10  # 0 is cancel; > 0 is activate

TIMEFRAME = TIMEFRAME_DICT[TIMEFRAME_TEXT]
START_STRING = str(START[0]) + str(START[1]).zfill(2) + str(START[2]).zfill(2) + str(START[3]).zfill(2) + str(START[4]).zfill(2)
END_STRING = str(END[0]) + str(END[1]).zfill(2) + str(END[2]).zfill(2) + str(END[3]).zfill(2) + str(END[4]).zfill(2)
CSV_FILE_PATH = MAIN_PATH + "//" + str(VERSION) + "//"
CSV_FILE_NAME = "T{}_{}_Frame-{}_Long-{}_Limit-{}_From-{}-to-{}_result.csv".format(
    DT_STRING,
    SYMBOL,
    str(TIMEFRAME_TEXT),
    str(LONG_MODE),
    str(LIMIT_UNIT),
    START_STRING,
    END_STRING)