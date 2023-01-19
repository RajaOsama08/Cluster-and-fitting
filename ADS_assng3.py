import pandas as pd
import numpy as np
import itertools as iter


def read_data(file):

    pd_df = pd.read_csv(file, skiprows=3)
    pd_df = pd_df.drop(["Unnamed: 66"], axis=1)
    return pd_df, pd_df.T


def exp(t, n0, g):

    t = t - 1960.0
    f = n0 * np.exp(g*t)
    print('exponential fun:', f)
    return f


def logt(t, n0, g, t0):
    f = n0 / (1 + np.exp(-g*(t - t0)))
    print('logistic:', f)
    return f


def norm(array):

    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    print('normalization:', scaled)
    return scaled


def norm_df(df, first=0, last=None):

    for col in df.columns[first:last]:
        df[col] = norm(df[col])
    print('df norm:', df[col])
    return df


def err_ranges(x, func, param, sigma):

    lower = func(x, *param)
    upper = lower

    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    print('upper limit:', upper, 'lower limit:', lower)
    return lower, upper


file = 'data.csv'
pd_df = read_data(file)
print(pd_df)
