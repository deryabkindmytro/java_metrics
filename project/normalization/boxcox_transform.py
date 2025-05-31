import numpy as np
from scipy.stats import boxcox

def normalize_data(df):
    transformed_data = df.copy()
    lambdas = {}
    for col in ['RFC', 'CBO', 'WMC']:
        transformed_data[col], lam = boxcox(df[col] + 1e-6)  # shift to avoid 0
        lambdas[col] = lam
    return transformed_data, lambdas