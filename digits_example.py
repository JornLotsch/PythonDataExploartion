#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:12:05 2022

@author: joern
"""
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.datasets import load_digits
from perform_projections_11 import perform_projections

digits = load_digits()
digits.data.shape


def plot_digits(data):
    fig, axes = plt.subplots(4,10, figsize = (10,4),
                             subplot_kw ={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):        
        ax.imshow(data[i].reshape(8,8), cmap = "binary", interpolation = "nearest", clim=(0,16))
        
#plot_digits(digits.data)

type(digits.data)        
dfDigits = pd.DataFrame(digits.data)
dfDigits.shape
dfDigits2 = dfDigits.to_numpy()

# model = PCA(0.5)
# components = model.fit_transform(digits.data)
# data_rec = model.inverse_transform(components)

implemented_ndimensionsmethods = ["KaiserGuttman", "ABCeigenvalues",
                                   "ExplainedVariance"]

for i, method in enumerate(implemented_ndimensionsmethods):
    print(method)
    
    PCA_Actual = perform_projections(data = dfDigits, scale = True, ndimensionsmethod=method)
    
    plot_digits(dfDigits2)
    
    digitsData_PCA_Actual_PCAimp = dfDigits.copy()
    digitsData_PCA_Actual_PCAimp.shape
    PCA_Actual_PCAimp_vars = list(PCA_Actual["relevant_features_pca"].index)
    for i in range(0,64):
        if i not in PCA_Actual_PCAimp_vars:
            digitsData_PCA_Actual_PCAimp[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_PCAimp.to_numpy())
    
    
    digitsData_PCA_Actual_relevant_features_reconstr_values = dfDigits.copy()
    digitsData_PCA_Actual_relevant_features_reconstr_values.shape
    PCA_Actual_relevant_features_reconstr_values_vars = list(PCA_Actual["relevant_features_reconstr_values"].index)
    for i in range(0,64):
        if i not in PCA_Actual_relevant_features_reconstr_values_vars:
            digitsData_PCA_Actual_relevant_features_reconstr_values[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_relevant_features_reconstr_values.to_numpy())
    
    
    digitsData_PCA_Actual_relevant_features_reconstr_dists = dfDigits.copy()
    digitsData_PCA_Actual_relevant_features_reconstr_dists.shape
    PCA_Actual_relevant_features_reconstr_dists_vars = list(PCA_Actual["relevant_features_reconstr_dists"].index)
    for i in range(0,64):
        if i not in PCA_Actual_relevant_features_reconstr_dists_vars:
            digitsData_PCA_Actual_relevant_features_reconstr_dists[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_relevant_features_reconstr_dists.to_numpy())
    
    digitsData_reconstructions = PCA_Actual["reconstructions"]
    plot_digits(digitsData_reconstructions)

