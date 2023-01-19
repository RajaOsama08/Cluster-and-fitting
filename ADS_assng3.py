import pandas as pd
import numpy as np


def read_data(file):

    pd_df = pd.read_csv(file, skiprows=3)
    pd_df = pd_df.drop(["Unnamed: 66"], axis=1)
    return pd_df, pd_df.T


file = 'data.csv'
pd_df = read_data(file)
print(pd_df)
