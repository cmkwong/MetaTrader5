from production.codes.views import pltView, printStat
from production.codes.controllers import mt5Controller
from production.codes import config
from production.codes.models import mt5Model
from production.codes.models.backtestModel import signalModel, returnModel, statModel


def moving_average_backtest(options, fast_index, slow_index, limit_unit, bins=100):
    with mt5Controller.Helper():
        df = mt5Model.get_historical_data(options['start'], options['end'], options['symbol'], options['timeframe'], options['timezone'])
        signal = signalModel.get_movingAverage_signal(df, fast_index, slow_index, limit_unit, long_mode=options['long_mode'],
                                                      backtest=options['backtest'])
        # information and statistic
        details = statModel.get_action_detail(df, signal)
        stat = statModel.get_stat(df, signal)
        stat["fast"], stat["slow"], stat["limit"] = fast_index, slow_index, limit_unit
        printStat.print_dict(details)
        printStat.print_dict(stat)

        # plot graph
        ret_list = returnModel.get_ret_list(df, signal)
        pltView.density(ret_list, bins=bins)

def optimize_moving_average(options, max_index=201):
    with mt5Controller.Helper() as helper:
        df = mt5Model.get_historical_data(options['start'], options['end'], options['symbol'], options['timeframe'], options['timezone'])
        for limit_unit in range(config.LIMIT_UNIT):
            for slow_index in range(1, max_index):
                for fast_index in range(1, slow_index):
                    if slow_index == fast_index:
                        continue
                    # moving average object
                    signal = signalModel.get_movingAverage_signal(df, fast_index, slow_index, limit_unit,
                                                                  long_mode=options['long_mode'], backtest=options['backtest'])
                    stat = statModel.get_stat(df, signal)
                    stat["fast"], stat["slow"], stat["limit"] = fast_index, slow_index, limit_unit
                    helper.append_dict_into_text(stat)
                    # print results
                    printStat.print_dict(stat)
        helper.write_csv()

options = {
    'start': config.START,
    'end': config.END,
    'symbol': config.SYMBOL,
    'timeframe': config.TIMEFRAME,
    'timezone': config.TIMEZONE,
    'long_mode': True,
    'backtest': True
}
moving_average_backtest(options, fast_index=3, slow_index=17, limit_unit=0, bins=100)
# optimize_moving_average(options)