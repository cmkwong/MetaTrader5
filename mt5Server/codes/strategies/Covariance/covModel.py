import pandas as pd
import numpy as np

def cov_matrix(array_2d, rowvar=False, bias=False):
    matrix = np.cov(array_2d, rowvar=rowvar, bias=bias)
    return matrix

def corela_matrix(array_2d, rowvar=False, bias=False):
    matrix = np.corrcoef(array_2d, rowvar=rowvar, bias=bias)
    return matrix

def corela_table(cor_matrix, symbol_list):
    cor_table = pd.DataFrame(cor_matrix, index=symbol_list, columns=symbol_list)
    return cor_table