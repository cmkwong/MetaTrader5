from production.codes.views import pltView
from production.codes.controllers import mt5Controller
from production.codes import config
from production.codes.models import mt5Model, returnModel, signalModel, statModel

slow_index, fast_index = 117, 106
limit_unit = 0

with mt5Controller.Helper() as tracker:
    df = mt5Model.get_historical_data(config.START, config.END, config.SYMBOL, config.TIMEFRAME, config.TIMEZONE)

    signal = signalModel.get_movingAverage_signal(df, slow_index, fast_index, limit_unit, long_mode=True, backtest=True)

    # information and statistic
    details = statModel.get_action_detail(df, signal)
    stat = statModel.get_stat(df, signal)
    stat["slow"], stat["fast"], stat["limit"] = slow_index, fast_index, limit_unit
    for key, value in details.items():
        print("{}: {:.5f}".format(key, value))
    for key, value in stat.items():
        print("{}: \t{:.5f}".format(key, value))

    # plot graph
    ret_list = returnModel.get_ret_list(df, signal)
    pltView.density(ret_list, bins=100)

    print()