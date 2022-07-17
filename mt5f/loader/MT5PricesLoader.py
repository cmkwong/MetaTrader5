import config
from backtest import exchgModel, pointsModel
from mt5f.loader import files
from mt5f.loader.BaseMT5PricesLoader import BaseMT5PricesLoader
from mt5f.mt5utils import segregation
import collections

from myUtils.paramType import SymbolList, DatetimeTuple, InputBoolean


# from dataclasses import dataclass, field
# from typing import List, Tuple, Dict
# from App.TkWidget import TkWidgetLabel
# from App.TkInitWidget import TkInitWidget


# mt5f loader price loader
class MT5PricesLoader(BaseMT5PricesLoader):  # created note 86a
    def __init__(self, all_symbol_info, data_path='', timezone='Hongkong', deposit_currency='USD'):
        self.all_symbol_info = all_symbol_info

        # for local
        self.data_path = data_path  # a symbol of loader that stored in this directory
        self.data_time_difference_to_UTC = config.DOWNLOADED_MIN_DATA_TIME_BETWEEN_UTC

        # property
        self.Prices = {}
        self.min_Prices = {}

        # for mt5f
        self.timezone = timezone
        self.deposit_currency = deposit_currency

        # prepare
        self.Prices_Collection = collections.namedtuple("Prices_Collection", ['o', 'h', 'l', 'c', 'cc', 'ptDv', 'quote_exchg', 'base_exchg', 'rawDfs'])
        self.latest_Prices_Collection = collections.namedtuple("latest_Prices_Collection", ['c', 'cc', 'ptDv', 'quote_exchg', 'rawDfs'])  # for latest Prices
        self._symbols_available = False  # only for usage of _check_if_symbols_available()

    def check_if_symbols_available(self, required_symbols, local):
        """
        check if symbols exist, note 83h
        :param required_symbols: [str]
        :param local: Boolean
        :return: None
        """
        if not self._symbols_available:
            for symbol in required_symbols:
                if not local:
                    try:
                        _ = self.all_symbol_info[symbol]
                    except KeyError:
                        raise Exception("The {} is not provided in this broker.".format(symbol))
                else:
                    fs = files.get_file_list(self.data_path)
                    if symbol not in fs:
                        raise Exception("The {} is not provided in my {}.".format(symbol, self.data_path))
            self._symbols_available = True

    def change_timeframe(self, df, timeframe='1H'):
        """
        note 84f
        :param df: pd.DataFrame, having header: open high low close
        :param rule: can '2H', https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling
        :return:
        """
        ohlc_rule = self._get_ohlc_rule(df)
        df = df.resample(timeframe).apply(ohlc_rule)
        df.dropna(inplace=True)
        return df

    def split_Prices(self, Prices, percentage):
        keys = list(Prices._asdict().keys())
        prices = collections.namedtuple("prices", keys)
        train_list, test_list = [], []
        for df in Prices:
            train, test = segregation.split_df(df, percentage)
            train_list.append(train)
            test_list.append(test)
        Train_Prices = prices._make(train_list)
        Test_Prices = prices._make(test_list)
        return Train_Prices, Test_Prices

    def get_Prices_format(self, symbols, prices, q2d_exchg_symbols, b2d_exchg_symbols):

        # get open prices
        open_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='100000')

        # get the change of close price
        close_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='000100')
        changes = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).fillna(0.0)

        # get the change of high price
        high_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='010000')

        # get the change of low price
        low_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='001000')

        # get point diff values
        # open_prices = _get_specific_from_prices(prices, symbols, ohlcvs='1000')
        points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, close_prices, close_prices.shift(periods=1), self.all_symbol_info)

        # get the quote to deposit exchange rate
        exchg_close_prices = self._get_specific_from_prices(prices, q2d_exchg_symbols, ohlcvs='000100')
        q2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "q2d")

        # get the base to deposit exchange rate
        exchg_close_prices = self._get_specific_from_prices(prices, b2d_exchg_symbols, ohlcvs='000100')
        b2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "b2d")

        # assign the column into each collection tuple
        Prices = self.Prices_Collection(o=open_prices,
                                        h=high_prices,
                                        l=low_prices,
                                        c=close_prices,
                                        cc=changes,
                                        ptDv=points_dff_values_df,
                                        quote_exchg=q2d_exchange_rate_df,
                                        base_exchg=b2d_exchange_rate_df,
                                        rawDfs=prices)

        return Prices

    def get_latest_Prices_format(self, symbols, prices, q2d_exchg_symbols, count):

        close_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='000100')
        if len(close_prices) != count:  # note 63a
            print("prices_df length of Data is not equal to count")
            return False

        # calculate the change of close price (with latest close prices)
        change_close_prices = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).fillna(0.0)

        # get point diff values with latest value
        points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, close_prices, close_prices.shift(periods=1), self.all_symbol_info)

        # get quote exchange with values
        exchg_close_prices = self._get_specific_from_prices(prices, q2d_exchg_symbols, ohlcvs='000100')
        q2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "q2d")
        # if len(q2d_exchange_rate_df_o) or len(q2d_exchange_rate_df_c) == 39, return false and run again
        if len(q2d_exchange_rate_df) != count:  # note 63a
            print("q2d_exchange_rate_df_o or q2d_exchange_rate_df_c length of Data is not equal to count")
            return False

        Prices = self.latest_Prices_Collection(c=close_prices,
                                               cc=change_close_prices,
                                               ptDv=points_dff_values_df,
                                               quote_exchg=q2d_exchange_rate_df,
                                               rawDfs=prices)

        return Prices

    def getPrices(self, *, symbols: SymbolList, start: DatetimeTuple, end: DatetimeTuple, timeframe: str, local: InputBoolean = False, latest: InputBoolean = False, count: int = 10, ohlcvs: str = '111100'):
        """
        :param local: if getting from local or from mt5f
        :param latest: if getting loader from past to now or from start to end
        """
        q2d_exchg_symbols = exchgModel.get_exchange_symbols(symbols, self.all_symbol_info, self.deposit_currency, 'q2d')
        b2d_exchg_symbols = exchgModel.get_exchange_symbols(symbols, self.all_symbol_info, self.deposit_currency, 'b2d')

        # read loader in dictionary format
        prices, min_prices = {}, {}
        required_symbols = list(set(symbols + q2d_exchg_symbols + b2d_exchg_symbols))
        self.check_if_symbols_available(required_symbols, local)  # if not, raise Exception
        if not latest:
            if local:
                min_prices = self._get_local_prices(self.data_path, required_symbols, self.data_time_difference_to_UTC, ohlcvs)
                # change the timeframe if needed
                if timeframe != '1min':  # 1 minute loader should not modify, saving the computation cost
                    for symbol in required_symbols:
                        prices[symbol] = self.change_timeframe(min_prices[symbol], timeframe)
                self.Prices, self.min_Prices = self.get_Prices_format(symbols, prices, q2d_exchg_symbols, b2d_exchg_symbols), self.get_Prices_format(symbols, min_prices, q2d_exchg_symbols, b2d_exchg_symbols)
            else:
                prices = self._get_mt5_prices(required_symbols, timeframe, self.timezone, start, end, ohlcvs, count)
                self.Prices = self.get_Prices_format(symbols, prices, q2d_exchg_symbols, b2d_exchg_symbols)
        else:
            prices = self._get_mt5_prices(required_symbols, timeframe, self.timezone, start, end, ohlcvs, count)
            self.Prices = self.get_latest_Prices_format(symbols, prices, q2d_exchg_symbols, count)

# @dataclass
# class get_data_TKPARAM(TkWidgetLabel):
#     symbols: dataclass = TkInitWidget(cat='get_data', id='1', type=TkWidgetLabel.DROPDOWN, value=['EURUSD', 'GBPUSD', 'USDJPY'])
#     start: Tuple[int] = field(default_factory=lambda: (2010, 1, 1))
#     end: Tuple[int] = field(default_factory=lambda: (2022, 1, 1))
#
#     def __init__(self):
#         super(get_data_TKPARAM, self).__init__()
#
# d = get_data_TKPARAM()
# print()
