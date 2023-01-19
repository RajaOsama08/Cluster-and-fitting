import pandas as pd
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as cluster
import seaborn as sns


def read_data(file):

    pd_df = pd.read_csv(file, skiprows=3)
    pd_df = pd_df.drop(["Unnamed: 66"], axis=1)
    return pd_df, pd_df.T

# model function taken form class lecture


def exp(t, n0, g):

    t = t - 1960.0
    f = n0 * np.exp(g*t)
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
    return scaled


def norm_df(df, first=0, last=None):

    for col in df.columns[first:last]:
        df[col] = norm(df[col])
    # print('df norm:', df[col])
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

    # print('upper limit:', upper, 'lower limit:', lower)
    return lower, upper


def fit_line(data, name, count_title, indicator1, indicator2, carbon_col,
             urban_col, y_label):
    '''
    this function will seprate data indicators and then plot graph show the
    good data fitting then it will show confidence of data and error uper and 
    lower limit
    '''
    # drop two column
    data = data.drop(['Country Code', 'Indicator Name'], axis=1)

    # fit exponential growth
    new_df = pd.DataFrame()

    # taking numpy array range
    year = np.arange(1963, 2020)
    val = 2e9, 0.04, 1990.0
    # print(year)
    data = data[data["Country Name"] == name]

    # matching indicators
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

    # urban population
    popt, covar = opt.curve_fit(
        logistic, new_df['Year'], new_df[urban_col], p0=(val))
    new_df["fit"] = logistic(new_df["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    fit = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)

    new_df.plot("Year", [urban_col, "fit"])
    plt.title(name + " Urban Population")
    plt.ylabel('Growth')
    plt.show()

    # curve fit for urban population
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

    # carbon emission plot
    popt, covar = opt.curve_fit(
        logistic, new_df['Year'], new_df[carbon_col], p0=(val))
    new_df["fit"] = logistic(new_df["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    fit = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)
    new_df.plot("Year", [carbon_col, "fit"])
    plt.title(name + count_title)
    plt.ylabel(y_label)
    plt.show()

    # curve fit for carbon emission
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


def distance(dist):
    dist = dist.loc[:, ['CO2', 'Urban']]
    # setting neighbor distance
    nbrs = NearestNeighbors(n_neighbors=6).fit(dist)

    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(dist)

    # sort the neighbor distances
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, 3]
    plt.plot(k_dist)
    plt.ylabel("distance")
    plt.xlabel("value")
    plt.show()


# taken from class lecture 9
def kmeans_clustring(data, title):
    '''
    this function will show the comparison of different kmeans cluster we we 
    used differenrt statistical methods and other tools

    '''

    df_ex = data[["CO2", "Urban"]]

    # Plot for four clusters
    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(df_ex)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    # Individual colours can be assigned to symbols. The label l is used to

    # plot the cluster graph
    plt.scatter(df_ex["CO2"], df_ex["Urban"], c=labels, cmap="Accent")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    for ic in range(4):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel("Co2")
    plt.ylabel("Urban population")
    plt.title(title)
    plt.show()


def map_corr(df, country, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns␣
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    df = df.loc[:, ['CO2', 'Urban']]
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_title(country)
    # Create correlation matrix
    corr_matrix = df.corr()
    # plot heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.9)
    sns.heatmap(corr_matrix,
                cmap='crest',
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                annot_kws=dict(size=12, weight='bold'),
                linecolor='black',
                linewidths=0.5,
                ax=ax)
    return plt.show()


if __name__ == "__main__":
    file = 'world_data.csv'
    data, transposed_data = read_data(file)
    dataframe = fit_line(data, "Pakistan", "Carbon Emission",
                         "EN.ATM.CO2E.LF.KT", "SP.URB.TOTL",
                         "CO2", "Urban", "CO2 growth")
    # print('fiti:', dataframe)

    distance(dataframe)
    kmeans_clustring(dataframe, "Pakistan")
    map_corr(dataframe, "Pakistan")

    dataframe2 = fit_line(data, "United States", "Carbon Emission",
                          "EN.ATM.CO2E.LF.KT", "SP.URB.TOTL",
                          "CO2", "Urban", "CO2 growth")
    distance(dataframe2)
    kmeans_clustring(dataframe2, "United States")
    map_corr(dataframe2, "United States")
