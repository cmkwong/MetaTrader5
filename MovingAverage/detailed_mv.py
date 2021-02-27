from production.codes.MovingAverage.lib import technical, graph_plot, common
from production.codes.MovingAverage.config import *
from production.codes.Trader import Connector, Data, Executor

slow_index, fast_index = 14, 10
limit_unit = 10

mt_data = Data.MetaTrader_Data(tz="Etc/UTC")
with common.Tracker(mt_data) as tracker:
    df = mt_data.get_historical_data(start=START, end=END, symbol=SYMBOL, timeframe=TIMEFRAME)

    movingAverage = technical.MovingAverage(df, long_mode=LONG_MODE, limit_unit=limit_unit)
    fast = movingAverage.get_moving_average(fast_index)
    slow = movingAverage.get_moving_average(slow_index)
    signal = movingAverage.get_signal(slow=slow, fast=fast)

    # information and statistic
    details = movingAverage.get_action_detail(signal)
    stat = movingAverage.get_ret_stat(signal, slow_index=slow_index, fast_index=fast_index)
    for key, value in details.items():
        print("{}: {:.5f}".format(key, value))
    for key, value in stat.items():
        print("{}: \t{:.5f}".format(key, value))

    # plot graph
    ret_list = movingAverage.get_ret_list(signal)
    graph_plot.density(ret_list, bins=100)

    print()