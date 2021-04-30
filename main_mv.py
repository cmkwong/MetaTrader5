from production.codes.controllers import mt5Controller
from production.codes import config
from production.codes.views import statView
from production.codes.models import mt5Model, signalModel, statModel

with mt5Controller.Helper() as tracker:
    df = mt5Model.get_historical_data(config.START, config.END, config.SYMBOL, config.TIMEFRAME, config.TIMEZONE)

    for limit_unit in range(config.LIMIT_UNIT):

        for slow_index in range(1, 201):

            for fast_index in range(1, slow_index):

                if slow_index == fast_index:
                    continue

                # moving average object
                signal = signalModel.get_movingAverage_signal(df, slow_index, fast_index, limit_unit, long_mode=True, backtest=True)

                stat = statModel.get_stat(df, signal)
                stat["slow"], stat["fast"], stat["limit"] = slow_index, fast_index, limit_unit
                tracker.append_dict_into_text(stat)

                # print results
                statView.print_dict(stat)

    tracker.write_csv()

