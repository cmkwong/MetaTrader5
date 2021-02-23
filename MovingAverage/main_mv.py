from production.codes.MovingAverage.lib import technical, data
from production.codes.common import server
from production.codes.MovingAverage.config import *

mt = server.MetaTrader_Connector()
with data.Tracker(mt) as tracker:
    df = mt.get_historical_data(start=START, end=END, symbol="USDJPY", timeframe=TIMEFRAME)
    movingAverage = technical.MovingAverage(SYMBOL, df, long_mode=LONG_MODE, limit_unit=LIMIT_UNIT)

    for slow_index in range(1, 201):

        slow = movingAverage.get_moving_average(slow_index)
        for fast_index in range(1, slow_index):

            if slow_index == fast_index:
                continue

            # moving average object
            fast = movingAverage.get_moving_average(fast_index)
            signal = movingAverage.get_signal(slow=slow, fast=fast)
            stat = movingAverage.get_ret_stat(signal, slow_index=slow_index, fast_index=fast_index)
            tracker.append_dict_into_text(stat)

            # print results
            print("\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
            for key, value in stat.items():
                print("{}:\t{:.5f}".format(key, value))

    tracker.write_csv()

