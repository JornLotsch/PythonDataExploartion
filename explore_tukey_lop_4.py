#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:47:48 2022

@author: joern
"""


def explore_tukey_lop(data, powers=[-2, -1, -0.5, 0, 0.5, 1, 2]):

    powersDict = {"reciprocal square": -2, "reciprocal": -1, "reciprocal square root": -
                  0.5, "log": 0, "square root": 0.5, "none": 1, "square": 2}
    if powers:
        if not all(x in powersDict.values() for x in powers):
            raise Warning("Input does not macth Tukey's powers!")
            return
        else:
            powers = {key: value for key, value in powersDict.items()
                      if value in powers}
    else:
        powers = powersDict

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
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

    fig, axes = plt.subplots(len(powers.keys()), 3, figsize=(20, 5*len(powers.keys())))
    for i, actual_power in enumerate(list(powers.values())):
        data_transformed = data_transform(data=data, tk_power=actual_power)
        data_transformed = data_transformed[~np.isnan(data_transformed)]
        ks_result = kstest(data_transformed, "norm")
        ks_result_p = str("{:.3E}".format(ks_result.pvalue))
        qq_values = qq_createValues(data_transformed)
        if len(powers.keys()) > 1:
            sns.histplot(ax=axes[i, 0], x=data_transformed).set(title="Histogram: Data = " + data.name)
            sns.distplot(ax=axes[i, 1], x=data_transformed, hist=False).set(title="pdf: transformation = " + list(powers.keys())[i])
            sns.scatterplot(ax=axes[i, 2], x=qq_values[:, 0], y=qq_values[:, 1]).set(title="QQ plot, KS-test p-value = " + ks_result_p)
            sns.lineplot(ax=axes[i, 2], x=qq_values[:, 0], y=qq_values[:, 2], color="salmon")
        else:
            sns.histplot(ax=axes[0], x=data_transformed).set(title="Histogram: Data = " + data.name)
            sns.distplot(ax=axes[1], x=data_transformed, hist=False).set(title="pdf: transformation = " + list(powers.keys())[i])
            sns.scatterplot(ax=axes[2], x=qq_values[:, 0], y=qq_values[:, 1]).set(title="QQ plot, KS-test p-value = " + ks_result_p)
            sns.lineplot(ax=axes[2], x=qq_values[:, 0], y=qq_values[:, 2], color="salmon")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Ordered data")

    return(fig)
