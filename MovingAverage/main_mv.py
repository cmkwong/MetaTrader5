from production.codes.MovingAverage.lib import technical, common
from production.codes.MovingAverage.config import *
from production.codes.Trader import Connector, Data, Executor

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
with common.Tracker(mt_data) as tracker:
    df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)
    for limit_unit in range(LIMIT_UNIT):
        movingAverage = technical.MovingAverage(df, long_mode=LONG_MODE, limit_unit=limit_unit)

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

