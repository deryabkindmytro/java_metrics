import numpy as np
import pandas as pd
from scipy.stats import f

def mahalanobis(x, mean, cov):
    delta = x - mean
    inv_cov = np.linalg.inv(cov)
    return float(np.dot(np.dot(delta.T, inv_cov), delta))

def evaluate_quality(raw_df, norm_df, models, lambdas):
    results = []
    data = norm_df[['RFC', 'CBO', 'WMC']]
    mean = data.mean()
    cov = data.cov()
    N = len(data)
    threshold = f.ppf(0.995, 3, N - 3) * 3 * (N - 1) / (N * (N - 3))

    for i in range(len(data)):
        row = data.iloc[i]
        md = mahalanobis(row, mean, cov)
        name = raw_df.iloc[i]['name']
        if md > threshold:
            quality = 'outlier — skip'
        else:
            inside_conf = []
            inside_pred = []
            for target, model in models.items():
                features = [f for f in ['RFC', 'CBO', 'WMC'] if f != target]
                X = row[features].to_frame().T
                y_pred = model.predict(X)[0]
                y_actual = row[target]
                error = abs(y_actual - y_pred)
                std_err = np.std(data[target] - model.predict(data[features]))
                inside_conf.append(error <= 2 * std_err)
                inside_pred.append(error <= 3 * std_err)

            if all(inside_conf):
                quality = 'medium'
            elif all(inside_pred):
                quality = 'high'
            else:
                quality = 'low'
        results.append({'name': name, 'quality': quality})
    return results
