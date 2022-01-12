#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:47:48 2022

@author: joern
"""


def explore_tukey_lop(data, once=True):

    powers = {"reciprocal square": -2, "reciprocal": -1, "reciprocal square root": -
              0.5, "log": 0, "square root": 0.5, "none": 1, "square": 2}
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import kstest

    def data_transform(data, tk_power):
        if tk_power != 0:
            a = np.power(data, tk_power)
        else:
            data_min = np.nanmin(data)
            if data_min <= 0:
                data = data - data_min + 1
            a = np.log(data)
        return(a)

    def qq_createValues(data):
        np.random.seed()
        qq = stats.probplot(data, dist="norm")
        qq_xval = [x[0] for x in qq][0]
        qq_yval = [x[1] for x in qq][0]
        qq_slope = [x[0] for x in qq][1]
        qq_intercept = [x[1] for x in qq][1]
        x_line = np.linspace(np.nanmin(qq_xval), np.nanmax(qq_xval), len(data))
        y_line = qq_slope*x_line+qq_intercept
        return np.vstack((qq_xval, qq_yval, x_line, y_line)).T

    def plot_tukey_lop(data, power_names, ks_result_p):
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        fig.suptitle("Data transformation: " + power_names)
        sns.histplot(ax=axes[0], x=data).set(
            title="Histogram: Data = " + data.name)
        sns.distplot(ax=axes[1], x=data, hist=False).set(title="pdf")
        qq_values = qq_createValues(data)
        sns.scatterplot(ax=axes[2], x=qq_values[:, 0], y=qq_values[:, 1]).set(
            title="QQ plot, KS-test p-value = " + ks_result_p)
        sns.lineplot(ax=axes[2], x=qq_values[:, 2],
                     y=qq_values[:, 3], color="salmon")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Ordered data")
        return(fig)

    if once == False:
        for i, actual_power in enumerate(list(powers.values())):
            data_transformed = data_transform(data=data, tk_power=actual_power)
            data_transformed = data_transformed[~np.isnan(data_transformed)]
            ks_result = kstest(data_transformed, "norm")
            ks_result_p = str("{:.2E}".format(ks_result.pvalue))
            plot_tukey_lop(data=data_transformed, power_names=list(
                powers.keys())[i], ks_result_p=ks_result_p)
    else:
        fig, axes = plt.subplots(len(powers.keys()), 3, figsize=(20, 30))
        for i, actual_power in enumerate(list(powers.values())):
            data_transformed = data_transform(data=data, tk_power=actual_power)
            data_transformed = data_transformed[~np.isnan(data_transformed)]
            ks_result = kstest(data_transformed, "norm")
            ks_result_p = str("{:.3E}".format(ks_result.pvalue))
            sns.histplot(ax=axes[i, 0], x=data_transformed).set(
                title="Histogram: Data = " + data.name)
            sns.distplot(ax=axes[i, 1], x=data_transformed, hist=False).set(
                title="pdf: transformation = " + list(powers.keys())[i])
            qq_values = qq_createValues(data_transformed)
            sns.scatterplot(ax=axes[i, 2], x=qq_values[:, 0], y=qq_values[:, 1]).set(
                title="QQ plot, KS-test p-value = " + ks_result_p)
            sns.lineplot(ax=axes[i, 2], x=qq_values[:, 2],
                         y=qq_values[:, 3], color="salmon")
            plt.xlabel("Normal theoretical quantiles")
            plt.ylabel("Ordered data")


explore_tukey_lop(data=data, once=True)
