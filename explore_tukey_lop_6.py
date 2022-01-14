#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:47:48 2022

@author: joern
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest, normaltest, zscore
from pandas.api.types import is_numeric_dtype


def explore_tukey_lop(data, powers=[-2, -1, -0.5, 0, 0.5, 1, 2], normtest="K^2 test", outlierremoval = False):

    powersDict = {"reciprocal square": -2, "reciprocal": -1, "reciprocal square root": -
                  0.5, "log": 0, "square root": 0.5, "none": 1, "square": 2}

    # %% Check some input parameters
    if powers:
        if not all(x in powersDict.values() for x in powers):
            raise Warning("Input does not macth Tukey's powers!")
            return
        else:
            powers = {key: value for key, value in powersDict.items()
                      if value in powers}
    else:
        powers = powersDict

    if not normtest in ["K^2 test", "KS test"]:
        raise Warning(
            "For normality testing, D’Agostino’s K^2 test and the KS test are implemented! Normailty test set to D’Agostino’s K^2 test.")
        normtest = "K^2 test"

    # %% Subfunction definitions
    def data_transform(data, tk_power):
        if tk_power != 0:
            a = np.power(data.astype("float"), tk_power)
        else:
            data_min = np.nanmin(data)
            if data_min <= 0:
                data = data - data_min + 1
            a = np.log(data.astype("float"))
        return(a)

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

    # %% Main function
    if not is_numeric_dtype(data):
        raise Warning("Data is not numeric")
        return
    else:
        data = data.dropna(axis = 0)
    
    if outlierremoval:
        data = data.dropna(axis = 0)
        z = np.abs(zscore(data))
        data = data[(z < 3)]
   
    if len(data) < 2:
       raise Warning("Too few data")
       return
    figColors = ["salmon" if x==1 else "Dodgerblue" for x in powers.values()]
    fig, axes = plt.subplots(
        len(powers.keys()), 3, figsize=(20, 5*len(powers.keys())))
    for i, actual_power in enumerate(list(powers.values())):
        data_transformed = data_transform(data=data, tk_power=actual_power)
        data_transformed = data_transformed[~np.isnan(data_transformed)]
        data_transformed = data_transformed[~np.isinf(data_transformed)]

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
        if len(powers.keys()) > 1:
            sns.histplot(ax=axes[i, 0], x=data_transformed, color=(figColors[i])).set(
                title="Histogram: Data = " + data.name)
            sns.distplot(ax=axes[i, 1], x=data_transformed, hist=False, color=(figColors[i])).set(
                title="pdf: transformation = " + list(powers.keys())[i])
            sns.scatterplot(ax=axes[i, 2], x=qq_values[:, 0], y=qq_values[:, 1], color=(figColors[i])).set(
                title="QQ plot, " + normtest + " p-value = " + normaltest_result_p)
            sns.lineplot(ax=axes[i, 2], x=qq_values[:, 0],
                         y=qq_values[:, 2], color="grey")
        else:
            sns.histplot(ax=axes[0], x=data_transformed, color=(figColors[i])).set(
                title="Histogram: Data = " + data.name)
            sns.distplot(ax=axes[1], x=data_transformed, hist=False, color=(figColors[i])).set(
                title="pdf: transformation = " + list(powers.keys())[i])
            sns.scatterplot(ax=axes[2], x=qq_values[:, 0], y=qq_values[:, 1], color=(figColors[i])).set(
                title="QQ plot, " + normtest + " p-value = " + normaltest_result_p)
            sns.lineplot(ax=axes[2], x=qq_values[:, 0],
                         y=qq_values[:, 2], color="grey")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Ordered data")
    return(fig)
