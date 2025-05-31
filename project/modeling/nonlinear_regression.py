from sklearn.linear_model import LinearRegression
import pandas as pd

def build_models(df):
    models = {}
    predictors = {
        'RFC': ['CBO', 'WMC'],
        'CBO': ['RFC', 'WMC'],
        'WMC': ['RFC', 'CBO']
    }
    for target, features in predictors.items():
        X = df[features]
        y = df[target]
        model = LinearRegression().fit(X, y)
        models[target] = model
    return models