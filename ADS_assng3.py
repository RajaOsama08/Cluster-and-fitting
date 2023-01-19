import pandas as pd
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
import scipy.optimize as opt


def read_data(file):

    pd_df = pd.read_csv(file, skiprows=3)
    pd_df = pd_df.drop(["Unnamed: 66"], axis=1)
    return pd_df, pd_df.T


def exp(t, n0, g):

    t = t - 1960.0
    f = n0 * np.exp(g*t)
    print('exponential fun:', f)
    return f


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
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


def fit_line(data, name, indicator, title, y_label):
    new_df = pd.DataFrame()
    year = np.arange(1963, 2020)
    print(year)
    data = data[data["Country Name"] == name]
    df_indicator = data[data["Indicator Code"] == indicator]

    df_indicator = df_indicator.drop(
        ["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis=1).T

    df_indicator = df_indicator.dropna()

    new_df['urban'] = df_indicator
    new_df['Year'] = pd.to_numeric(year)

    popt, covar = opt.curve_fit(
        logistic, new_df['Year'], new_df['urban'], p0=(2e9, 0.05, 1990.0))

    new_df["fit"] = logistic(new_df["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1963, 2040)
    forecast = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)

    new_df.plot("Year", ["urban", "fit"])
    plt.title(str(name)+title)
    plt.ylabel(y_label)
    plt.show()

    plt.figure()
    plt.plot(new_df["Year"],
             new_df["urban"], label="Urban")
    plt.title(str(name)+title)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="magenta", alpha=0.6)
    plt.xlabel("year")
    plt.ylabel("Urban Population")
    plt.legend()
    plt.show()
    return new_df


file = 'world_data.csv'
pd_df, pd_df_trans = read_data(file)

fit_line(pd_df, 'Pakistan', 'EN.ATM.CO2E.LF.KT', 'Forest area', 'coal growth')
# fit_line(pd_df, 'Pakistan', 'BX.KLT.DINV.WD.GD.ZS')
