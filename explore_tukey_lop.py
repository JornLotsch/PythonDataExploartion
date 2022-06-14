#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:47:48 2022

@author: joern
"""

# %% imports

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest, normaltest, zscore, boxcox
from pandas.api.types import is_numeric_dtype


# %%
def clean_data(data, outlierremoval):
    type(data)

    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    if not is_numeric_dtype(data):
        raise Warning("Data is not numeric.")
        return
    else:
        data = data[~np.isnan(data)]

    # if len(data.value_counts()) == 2:
    #     raise Warning("Data seems to be binary.")
    #     return

    # if len(data.value_counts()) == 1:
    #     raise Warning("All instances have the same value.")
    #     return

    if outlierremoval:
        z = np.abs(zscore(data))
        data = data[(z < 3)]

    if len(data) < 2:
        raise Warning("Too few data.")
        return

    dataOK = True

    return data, dataOK

# %% Data transformation


def data_transform(data, tk_power):
    BClambda = -99
    if tk_power != 0:
        if tk_power != 10:
            a = np.power(data.astype("float"), tk_power)
        else:
            data_min = np.nanmin(data)
            if data_min <= 0:
                data = data - data_min + 1
            a, BClambda = boxcox(data.astype("float"))
    else:
        data_min = np.nanmin(data)
        if data_min <= 0:
            data = data - data_min + 1
        a = np.log(data.astype("float"))
    return a, BClambda

# %% QQ


def qq_createValues(data):
    np.random.seed()
    sample_size = len(data)
    qq = np.ones([sample_size, 3])
    qq[:, 0] = np.sort(np.random.normal(size=sample_size))
    qq[:, 1] = np.sort(data[0:sample_size])
    model = LinearRegression()
    X = qq[:, 0][:, np.newaxis]
    model.fit(X, qq[:, 1])
    predictions = model.predict(X)
    qq[:, 2] = predictions
    return(qq)

# %% Plots


def create_plots(data, powers, normtest):
    BClambda = 1
    n_transformations = len(powers.keys())
    fig, axes = plt.subplots(n_transformations, 3,
                             figsize=(20, 5*n_transformations))
    for i, actual_power in enumerate(list(powers.values())):
        data_transformed, BClambdaX = data_transform(
            data=data, tk_power=actual_power)
        if actual_power == 10:
            BClambda = BClambdaX
        data_transformed = data_transformed[~np.isnan(data_transformed)]
        data_transformed = data_transformed[~np.isinf(data_transformed)]

        figColors = [("purple" if i == round(BClambda, 0) else (
            "salmon" if i == 1 else "dodgerblue")) for i in powers.values()]

        if normtest == "KS test":
            data_transformed_standardized = preprocessing.scale(
                data_transformed)
            normaltest_result = kstest(
                data_transformed_standardized, "norm")
            normaltest_result_p = str(
                "{:.3E}".format(normaltest_result.pvalue))
        else:
            normaltest_result = normaltest(data_transformed)
            normaltest_result_p = str(
                "{:.3E}".format(normaltest_result.pvalue))

        qq_values = qq_createValues(data=data_transformed)

        if actual_power == 1 and normaltest_result.pvalue > 0.05:
            figColors[i] = "red"
        if len(powers.keys()) > 1:
            sns.histplot(ax=axes[i, 0], x=data_transformed, color=(figColors[i])).set(
                title="Histogram: Data = " + data.name)
            pdf_title = "pdf: transformation = " + list(powers.keys())[i] if actual_power != 10 else \
                "pdf: transformation = " + \
                list(powers.keys())[i] + ", lambda = " + \
                "{:.2f}".format(round(BClambda, 2))
            sns.kdeplot(ax=axes[i, 1], x=data_transformed, color=(figColors[i])).set(
                title=pdf_title)

            sns.scatterplot(ax=axes[i, 2], x=qq_values[:, 0], y=qq_values[:, 1], color=(figColors[i])).set(
                title="QQ plot, " + normtest + " p-value = " + normaltest_result_p)
            sns.lineplot(ax=axes[i, 2], x=qq_values[:, 0],
                         y=qq_values[:, 2], color="grey")
        else:
            sns.histplot(ax=axes[0], x=data_transformed, color=(figColors[i])).set(
                title="Histogram: Data = " + data.name)
            sns.kdeplot(ax=axes[1], x=data_transformed, color=(figColors[i])).set(
                title="pdf: transformation = " + list(powers.keys())[i])
            sns.scatterplot(ax=axes[2], x=qq_values[:, 0], y=qq_values[:, 1], color=(figColors[i])).set(
                title="QQ plot, " + normtest + " p-value = " + normaltest_result_p)
            sns.lineplot(ax=axes[2], x=qq_values[:, 0],
                         y=qq_values[:, 2], color="grey")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Ordered data")

    return fig


# %% Main function
def explore_tukey_lop(data, powers=[-3, -2, -1, -0.5, -0.333333, 0, 0.333333, 0.5, 1, 2, 3], normtest="K^2 test", outlierremoval=False):

    powersDict = {"BoxCox": 10, "reciprocal cube": -3, "reciprocal square": -2, "reciprocal": -1, "reciprocal square root": -
                  0.5, "reciprocal cube root": -0.333333, "log": 0, "cube root": 0.333333, "square root": 0.5, "none": 1, "square": 2, "cube": 3}

    if powers:
        if not all(x in powersDict.values() for x in powers):
            raise Warning("Input does not macth Tukey's powers.")
            return
        else:
            powers = {key: value for key, value in powersDict.items()
                      if value == 10 or value in powers}
    else:
        powers = powersDict

    if not normtest in ["K^2 test", "KS test"]:
        print("For normality testing, D’Agostino’s K^2 test and the KS test are implemented! Normailty test set to D’Agostino’s K^2 test.")
        normtest = "K^2 test"

    CleanedData, dataOK = clean_data(data, outlierremoval)

    if dataOK:
        figure = create_plots(
            data=CleanedData, powers=powers, normtest=normtest)

    return figure
