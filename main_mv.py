from production.codes.lib import common, MovingAverage
from production.codes.lib.config import *
from production.codes.lib.Trader import Data

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
with common.Tracker(mt_data) as tracker:

    df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)

    for limit_unit in range(LIMIT_UNIT):

        movingAverage = MovingAverage.MovingAverage(df, backtest=True, long_mode=LONG_MODE)
        # max_limit = MaxLimitClosed.MaxLimitClosed(limit_unit=limit_unit)

        for slow_index in range(1, 201):

            slow = movingAverage.get_moving_average(slow_index)
            for fast_index in range(1, slow_index):

                if slow_index == fast_index:
                    continue

                # moving average object
                fast = movingAverage.get_moving_average(fast_index)
                signal = movingAverage.get_signal(slow=slow, fast=fast, limit_unit=limit_unit)
                # signal = max_limit.modify(signal)

                stat = movingAverage.get_stat(signal)
                stat["slow"], stat["fast"], stat["limit"] = slow_index, fast_index, limit_unit
                tracker.append_dict_into_text(stat)

                # print results
                print("\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
                for key, value in stat.items():
                    print("{}:\t{:.5f}".format(key, value))

    tracker.write_csv()

