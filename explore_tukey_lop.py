#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:47:48 2022

@author: joern
"""



data = dfRiechenVerduennungAbstand.BMI * 1
data[12]= -11
def explore_tukey_lop(data, once = True):
    power = [-2, -1, -0.5, 0, 0.5, 1, 2]
    power_names = ["reciprocal square", "reciprocal", "reciprocal square root", "log", "square root", "none", "square"]
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import kstest
    from sklearn.linear_model import LinearRegression

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
        sample_size = len(data)
        qq = np.ones([sample_size, 2])
        np.random.seed()
        qq[:, 1] = np.sort(data[0:sample_size])
        qq[:, 0] = np.sort(np.random.normal(size = sample_size))
        return qq

    def qq_createLine(qq_values):
        sample_size = len(qq_values[:,0])
        qq = np.ones([sample_size, 2])
        model = LinearRegression()
        X = qq_values[:,0][:, np.newaxis]
        model.fit(X, qq_values[:,1])
        predictions = model.predict(X)
        qq[:, 0] = qq_values[:,0]
        qq[:, 1] = predictions
        return(qq)
    
    def qq_plot2(data):
        qq = stats.probplot(data, dist="norm")
        xval = [x[0] for x in qq][0]
        yval = [x[1] for x in qq][0]
        #slope = [x[1] for x in qq][1]
        #intercept = [x[1] for x in qq][1]
        return np.vstack((xval, yval)).T

    def plot_tukey_lop(data, power_names, ks_result_p):
        fig, axes = plt.subplots(1, 3,figsize=(15, 10))
        fig.suptitle("Data transformation: " + power_names)
        sns.histplot(ax=axes[0], x = data).set(title = "Histogram: Data = " + data.name)
        sns.distplot(ax=axes[1], x = data, hist = False).set(title = "pdf")

        #qqplot = stats.probplot(data, dist="norm", plot = plt)
        qq_values = qq_createValues(data)
        qq_xval = qq_values[:,0]
        qq_yval = qq_values[:,1]
        qq_line = qq_createLine(qq_values)
        sns.scatterplot(ax=axes[2], x = qq_xval, y = qq_yval).set(title = "QQ plot, KS-test p-value = " + ks_result_p)
        sns.lineplot(ax=axes[2], x = qq_line[:,0], y = qq_line[:,1], color = "salmon")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Ordered data")
        return(fig)

    if once == False:
         for i, actual_power in enumerate(power):
             data_transformed = data_transform(data = data, tk_power = actual_power)
             data_transformed = data_transformed[~np.isnan(data_transformed)]
             ks_result = kstest(data_transformed, "norm")
             ks_result_p = str("{:.2E}".format(ks_result.pvalue))
             plot_tukey_lop(data = data_transformed, power_names = power_names[i], ks_result_p = ks_result_p)
    else:
        fig, axes = plt.subplots(len(power), 3,figsize=(20, 30))
        for i, actual_power in enumerate(power):
            data_transformed = data_transform(data = data, tk_power = actual_power)
            data_transformed = data_transformed[~np.isnan(data_transformed)]
            ks_result = kstest(data_transformed, "norm")
            ks_result_p = str("{:.2E}".format(ks_result.pvalue))
            sns.histplot(ax=axes[i,0], x = data_transformed).set(title = "Histogram: Data = " + data.name)
            sns.distplot(ax=axes[i,1], x = data_transformed, hist = False).set(title = "pdf: transformation = " + power_names[i])
            qq_values = qq_createValues(data_transformed)
            qq_xval = qq_values[:,0]
            qq_yval = qq_values[:,1]
            qq_line = qq_createLine(qq_values)
            sns.scatterplot(ax=axes[i,2], x = qq_xval, y = qq_yval).set(title = "QQ plot, KS-test p-value = " + ks_result_p)
            sns.lineplot(ax=axes[i,2], x = qq_line[:,0], y = qq_line[:,1], color = "salmon")
            plt.xlabel("Normal theoretical quantiles")
            plt.ylabel("Ordered data")

explore_tukey_lop(data = data, once = True)


