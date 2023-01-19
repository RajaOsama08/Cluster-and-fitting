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
    """Calculates the logistic function with scale factor n0 and
    growth rate g"""
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


def fit_line(data, name, count_title, indicator1, indicator2, carbon_col, urban_col, y_label):
    '''
    this  function will seprate data indicators and then plot graph show the good data fitting 
    then it will show confidence of data and error uper and lower limit
    '''
    # drop column
    data = data.drop(['Country Code', 'Indicator Name'], axis=1)
    # fit exponential growth
    new_df = pd.DataFrame()
    year = np.arange(1963, 2020)
    print(year)
    data = data[data["Country Name"] == name]

    carbon_data = data[data["Indicator Code"]
                       == indicator1]

    urban_data = data[data["Indicator Code"] == indicator2]

    # drop country and indicator column to plot data
    carbon_data = carbon_data.drop(
        ["Country Name", "Indicator Code"], axis=1).T
    urban_data = urban_data.drop(
        ["Country Name", "Indicator Code"], axis=1).T

    carbon_data = carbon_data.dropna()
    urban_data = urban_data.dropna()

    new_df[carbon_col] = carbon_data
    new_df[urban_col] = urban_data
    new_df['Year'] = pd.to_numeric(year)

    # curve fit for urban population
    popt, covar = opt.curve_fit(
        logistic, new_df['Year'], new_df[urban_col], p0=(2e9, 0.04, 1990.0))
    new_df["fit"] = logistic(new_df["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1963, 2020)
    fit = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)

    new_df.plot("Year", [urban_col, "fit"])
    plt.title(name + " Urban Population")
    plt.ylabel('Growth')
    plt.show()

    plt.figure()
    plt.plot(new_df["Year"],
             new_df[urban_col], label="Urban")
    plt.title(name + " Urban Population Fit")
    plt.plot(year, fit, label="curver fit")
    plt.fill_between(year, low, up, color="magenta", alpha=0.6)
    plt.xlabel("year")
    plt.ylabel("Growth")
    plt.legend()
    plt.show()

    # curve fit for carbon emission
    popt, covar = opt.curve_fit(
        logistic, new_df['Year'], new_df[carbon_col], p0=(2e9, 0.04, 1990.0))
    new_df["fit"] = logistic(new_df["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    fit = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)
    new_df.plot("Year", [carbon_col, "fit"])
    plt.title(name + count_title)
    plt.ylabel(y_label)
    plt.show()

    plt.figure()
    plt.plot(new_df["Year"], new_df[carbon_col], label="co2")
    plt.title(name + " " + count_title + " " + "Fit")
    plt.plot(year, fit, label="curve fit")
    plt.fill_between(year, low, up, color="magenta", alpha=0.6)
    plt.xlabel("year")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    return new_df


if __name__ == "__main__":
    file = 'world_data.csv'
    data, transposed_data = read_data(file)
    fiti = fit_line(data, "Pakistan", "Carbon Emission",
                    "EN.ATM.CO2E.LF.KT", "SP.URB.TOTL",
                    "CO2", "Urban", "CO2 growth")
    fiti = fit_line(data, "United States", "Carbon Emission",
                    "EN.ATM.CO2E.LF.KT", "SP.URB.TOTL",
                    "CO2", "Urban", "CO2 growth")
