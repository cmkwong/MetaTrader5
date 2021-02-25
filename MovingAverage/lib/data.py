from production.codes.MovingAverage.config import *
import MetaTrader5 as mt5

class Tracker:
    def __init__(self, mt):
        self.mt = mt
        self.text = ''
        self.text_line = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        mt5.shutdown()
        print("MetaTrader Shutdown.")

    def append_dict_into_text(self, dict):
        if self.text_line == 0:
            for key in dict.keys():
                self.text += key + ','
            index = self.text.rindex(',')           # find the last index
            self.text = self.text[:index] + '\n'    # and replace
            self.text_line += 1
        for value in dict.values():
            self.text += str(value) + ','
        index = self.text.rindex(',')           # find the last index
        self.text = self.text[:index] + '\n'    # and replace
        self.text_line += 1

    def write_csv(self):
        print("\nFrame: {}\nLong Mode: {}\nFrom: {}\nTo: {}\n".format(str(TIMEFRAME_TEXT), str(LONG_MODE), START_STRING, END_STRING))
        print("Writing csv ... ", end='')
        with open(CSV_FILE_PATH + CSV_FILE_NAME, 'w') as f:
            f.write(self.text)
        print("OK")