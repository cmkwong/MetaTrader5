from production.codes.lib import common, graph_plot, MovingAverage
from production.codes.lib.config import *
from production.codes.lib.Trader import Data

slow_index, fast_index = 117, 106
limit_unit = 0

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
with common.Tracker(mt_data) as tracker:
    df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)

    movingAverage = MovingAverage.MovingAverage(df, backtest=True, long_mode=LONG_MODE)
    # max_limit = MaxLimitClosed.MaxLimitClosed(limit_unit=limit_unit)

    fast = movingAverage.get_moving_average(fast_index)
    slow = movingAverage.get_moving_average(slow_index)
    signal = movingAverage.get_signal(slow=slow, fast=fast, limit_unit=limit_unit)
    # signal = max_limit.modify(signal)

    # information and statistic
    details = movingAverage.get_action_detail(signal)
    stat = movingAverage.get_stat(signal)
    stat["slow"], stat["fast"], stat["limit"] = slow_index, fast_index, limit_unit
    for key, value in details.items():
        print("{}: {:.5f}".format(key, value))
    for key, value in stat.items():
        print("{}: \t{:.5f}".format(key, value))

    # plot graph
    ret_list = movingAverage.get_ret_list(signal)
    graph_plot.density(ret_list, bins=100)

    print()