from production.codes.MovingAverage.lib import technical, graph_plot, data
from production.codes.common import server
from production.codes.MovingAverage.config import *

slow_index, fast_index = 68, 26
mt = server.MetaTrader_Connector()
with data.Tracker(mt) as tracker:
    df = mt.get_historical_data(start=START, end=END, symbol="USDJPY", timeframe=TIMEFRAME)

    movingAverage = technical.MovingAverage(SYMBOL, df)
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