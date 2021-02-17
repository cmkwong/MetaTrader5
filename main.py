from production.codes.lib import data, technical, graph_plot
import MetaTrader5 as mt5

mt = data.MetaTrader_Connector()
with data.Tracker(mt) as tracker:
    df = mt.get_historical_data(start=(2021,1,1,0,0), end=(2021,2,1,0,0), symbol="USDJPY", timeframe=mt5.TIMEFRAME_M10)
    # print(df)
    # print(len(df))
    # last_tick_info = mt.get_last_tick(symbol="USDJPY")
    # print(last_tick_info['bid'])
    # symbols = mt.get_symbols()

    # moving average object
    slow_index, fast_index = 200, 1
    movingAverage = technical.MovingAverage("USDJPY", df)
    fast = movingAverage.get_moving_average(fast_index)
    slow = movingAverage.get_moving_average(slow_index)
    signal = movingAverage.get_signal(slow=slow, fast=fast)

    # information and statistic
    details = movingAverage.get_signal_detail(signal)
    stat = movingAverage.get_ret_stat(signal, slow_index=slow_index, fast_index=fast_index)
    for key, value in details.items():
        print("{}: {:.5f}".format(key, value))
    for key, value in stat.items():
        print("{}: \t{:.5f}".format(key, value))

    # plot graph
    ret_list = movingAverage.get_ret_list(signal)
    graph_plot.density(ret_list,bins=100)

    print()