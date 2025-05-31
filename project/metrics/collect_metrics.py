import pandas as pd

def load_metrics(filepath):
    df = pd.read_csv(filepath)
    return df  # expects columns: name, RFC, CBO, WMC