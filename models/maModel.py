import pandas as pd
from production.codes.views import printStat
from production.codes import config
from production.codes.models import mt5Model
from production.codes.models.backtestModel import signalModel,statModel, techModel

def find_optimize_moving_average(options, max_index=201):
    with mt5Model.Helper() as helper:
        df = mt5Model.get_historical_data(options['symbol'], options['timeframe'], options['timezone'],
                                          options['start'], options['end'])
        for limit_unit in range(config.LIMIT_UNIT):
            for slow_index in range(1, max_index):
                for fast_index in range(1, slow_index):
                    if slow_index == fast_index:
                        continue
                    # moving average object
                    signal = signalModel.get_movingAverage_signal(df, fast_index, slow_index, limit_unit, options['long_mode'])
                    stat = statModel.get_stat(df, signal)
                    stat["fast"], stat["slow"], stat["limit"] = fast_index, slow_index, limit_unit
                    helper.append_dict_into_text(stat)
                    # print results
                    printStat.print_dict(stat)
        helper.write_csv()

def get_ma_data(close_prices, fast_param, slow_param):
    ma_data = pd.DataFrame(index=close_prices.index)
    ma_data['fast'] = techModel.get_moving_average(close_prices, fast_param)
    ma_data['slow'] = techModel.get_moving_average(close_prices, slow_param)
    return ma_data
