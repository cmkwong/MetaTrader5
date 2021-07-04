from production.codes.models.backtestModel import indexModel, pointsModel, exchgModel
from production.codes.models import coinModel
import pandas as pd
import numpy as np

class Calculator:
    def __init__(self, symbols, all_symbols_info, Prices, coefficient_vector, signal, long_mode, min_Prices=None, slsp=None, lot_times=1):

        self.symbols = symbols
        self.all_symbols_info = all_symbols_info

        self.Prices = Prices
        self.min_Prices = min_Prices
        self.coefficient_vector = coefficient_vector
        self.signal = signal
        self.long_mode = long_mode
        self.slsp = slsp
        self.lot_times = lot_times

    def calculate(self):
        # prepare
        self.modified_coefficient_vector = coinModel.get_modified_coefficient_vector(self.coefficient_vector, self.long_mode, self.lot_times)
        self.modify_exchg_q2d = exchgModel.get_exchg_by_signal(self.Prices.quote_exchg, self.signal)
        # calculate required values
        self.ret, self.earning = self.get_ret_earning()
        self.ret_list, self.earning_list = self.get_ret_earning_list()
        self.ret_by_signal, self.earning_by_signal = self.get_ret_earning_by_signal()
        self.accum_ret, self.accum_earning = self.get_accum_ret_earning()
        self.total_ret, self.total_earning = self.get_total_ret_earning()

    def get_ret_earning(self):

        # prepare
        new_prices, old_prices = self.Prices.o, self.Prices.o.shift(1)
        points_dff_values_df = pointsModel.get_points_dff_values_df(self.symbols, new_prices, old_prices, self.all_symbols_info)

        # ret
        change = (new_prices - old_prices) / old_prices
        olds = np.sum(np.abs(self.modified_coefficient_vector))
        news = (np.abs(self.modified_coefficient_vector) + (change * self.modified_coefficient_vector)).sum(axis=1)
        ret = pd.Series(news / olds, index=new_prices.index, name="return")

        # earning
        weighted_pt_diff = points_dff_values_df.values * self.modified_coefficient_vector.reshape(-1, )
        # calculate the price in required deposit dollar
        earning = pd.Series(np.sum(self.modify_exchg_q2d.values * weighted_pt_diff, axis=1), index=self.modify_exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)

        return ret, earning

    def get_ret_earning_by_signal(self):
        """
        :param ret: pd.Series
        :param earning: earning
        :param signal: pd.Series
        :param slsp: tuple(stop loss (negative), stop profit (positive))
        :return: pd.Series
        """
        ret_by_signal = pd.Series(self.signal.shift(2).values * self.ret.values, index=self.signal.index, name="ret_by_signal").fillna(1.0).replace({0: 1})
        earning_by_signal = pd.Series(self.signal.shift(2).values * self.earning.values, index=self.signal.index, name="earning_by_signal").fillna(0.0)  # shift 2 unit see (30e)
        if self.slsp != None:
            start_index, end_index = indexModel.get_action_start_end_index(self.signal)
            for raw_s, raw_e in zip(start_index, end_index):
                s, e = indexModel.get_required_index(ret_by_signal, raw_s, step=1), indexModel.get_required_index(ret_by_signal, raw_e, step=0) # why added 1, see notes (6) // Why step=0, note 87b
                ret_by_signal.loc[s:e], earning_by_signal.loc[s:e] = modify_ret_earning_with_SLSP(self.ret.loc[s:e], self.earning.loc[s:e], self.slsp[0], self.slsp[1])
        return ret_by_signal, earning_by_signal

    def get_ret_earning_list(self):
        start_index, end_index = indexModel.get_action_start_end_index(self.signal)
        rets, earnings = [], []
        for raw_s, raw_e in zip(start_index, end_index):
            s, e = indexModel.get_required_index(self.ret, raw_s, step=1), indexModel.get_required_index(self.ret, raw_e, step=0)  # why added 1, see notes (6) // Why step=0, note 87b
            ret_series, earning_series = self.ret.loc[s:e], self.earning.loc[s:e]
            if self.slsp != None:
                ret_series, earning_series = modify_ret_earning_with_SLSP(ret_series, earning_series, self.slsp[0], self.slsp[1])  # modify the return and earning if has stop-loss and stop-profit setting
            rets.append(ret_series.prod())
            earnings.append(np.sum(earning_series))
        return rets, earnings

    def get_total_ret_earning(self):
        """
        :return: float, float
        """
        total_ret, total_earning = 1, 0
        for ret, earning in zip(self.ret_list, self.earning_list):
            total_ret *= ret
            total_earning += earning
        return total_ret, total_earning

    def get_accum_ret_earning(self):
        """
        :param ret: pd.Series
        :param earning: pd.Series
        :param signal: pd.Series
        :param slsp: tuple(stop loss (negative), stop profit (positive))
        :return: accum_ret (pd.Series), accum_earning (pd.Series)
        """
        accum_ret = pd.Series(self.ret_by_signal.cumprod(), index=self.ret_by_signal.index, name="accum_ret")  # Simplify the function note 47a
        accum_earning = pd.Series(self.earning_by_signal.cumsum(), index=self.ret_by_signal.index, name="accum_earning")  # Simplify the function note 47a
        return accum_ret, accum_earning

def modify_ret_earning_with_SLSP_late(ret_series, earning_series, sl, sp):
    """
    equation see 77ab
    :param ret_series: pd.Series with numeric index
    :param earning_series: pd.Series with numeric index
    :param sl: stop-loss (negative value)
    :param sp: stop-profit (positive value)
    :return: ret (np.array), earning (np.array)
    """
    total = 0
    ret_mask, earning_mask = np.ones((len(ret_series),)), np.zeros((len(ret_series),))
    for i, (r, e) in enumerate(zip(ret_series, earning_series)):
        total += e
        ret_mask[i], earning_mask[i] = ret_series[i], earning_series[i]
        if total >= sp:
            break
        elif total <= sl:
            break
    return ret_mask, earning_mask

def modify_ret_earning_with_SLSP(ret_series, earning_series, sl, sp):
    """
    equation see 49b
    :param ret_series: pd.Series with numeric index
    :param earning_series: pd.Series with numeric index
    :param sl: stop-loss (negative value)
    :param sp: stop-profit (positive value)
    :return: ret (np.array), earning (np.array)
    """
    total = 0
    sl_buffer, sp_buffer = sl, sp
    ret_mask, earning_mask = np.ones((len(ret_series),)), np.zeros((len(ret_series),))
    for i, (r, e) in enumerate(zip(ret_series, earning_series)):
        total += e
        if total >= sp:
            ret_mask[i] = 1 + ((r - 1) / e) * sp_buffer
            earning_mask[i] = sp_buffer
            break
        elif total <= sl:
            ret_mask[i] = 1 - ((1 - r) / e) * sl_buffer
            earning_mask[i] = sl_buffer
            break
        else:
            ret_mask[i], earning_mask[i] = ret_series[i], earning_series[i]
            sl_buffer -= e
            sp_buffer -= e
    return ret_mask, earning_mask

def get_value_of_ret_earning(symbols, new_values, old_values, q2d_at, all_symbols_info, lot_times, coefficient_vector, long_mode):
    """
    This is calculate the return and earning from raw value (instead of from dataframe)
    :param symbols: [str]
    :param new_values: np.array (Not dataframe)
    :param old_values: np.array (Not dataframe)
    :param q2d_at: np.array, values at brought the assert
    :param coefficient_vector: np.array
    :param all_symbols_info: nametuple
    :param long_mode: Boolean
    :return: float, float: ret, earning
    """

    modified_coefficient_vector = coinModel.get_modified_coefficient_vector(coefficient_vector, long_mode, lot_times)

    # ret value
    changes = (new_values - old_values) / old_values
    olds = np.sum(np.abs(modified_coefficient_vector))
    news = (np.abs(modified_coefficient_vector) + (changes * modified_coefficient_vector)).sum()
    ret = news / olds

    # earning value
    points_dff_values = pointsModel.get_points_dff_values(symbols, new_values, old_values, all_symbols_info)
    weighted_pt_diff = points_dff_values * modified_coefficient_vector.reshape(-1, )
    # calculate the price in required deposit dollar
    earning = np.sum(q2d_at * weighted_pt_diff)

    return ret, earning

