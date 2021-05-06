import numpy as np
from statsmodels.tsa.stattools import adfuller
import collections
import pandas as pd

def z_col(col):
    mean = np.mean(col)
    std = np.std(col)
    normalized_col = (col - mean) / std
    return normalized_col

def z_score(x):
    z = (x - np.mean(x)) / np.std(x)
    return z

def z_score_with_rolling_mean(x, window):
    rolling_mean = pd.Series(x).rolling(window).mean()
    z = (x - rolling_mean) / np.std(rolling_mean)
    return z

def perform_ADF_test(x):
    adf_result = collections.namedtuple("adf_result", ["test_statistic", "pvalue", "critical_values"])
    result = adfuller(x)
    adf_result.test_statistic = result[0]
    adf_result.pvalue = result[1]
    adf_result.critical_values = result[4]
    return adf_result