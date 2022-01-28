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
    fig, axes = plt.subplots(2,10, figsize = (10,2),
                             subplot_kw ={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):        
        ax.imshow(data[i].reshape(8,8), cmap = "binary", interpolation = "nearest", clim=(0,16))
        
def plot_digits4(data1, data2, data3, data4):
    fig, axes = plt.subplots(8,10, figsize = (10,8),
                             subplot_kw ={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):
        
        if i < 20: 
            data = data1
        elif i < 40:
            data = data2
        elif i < 60:
            data = data3
        else:
            data = data4
            
        ax.imshow(data[i].reshape(8,8), cmap = "binary", interpolation = "nearest", clim=(0,16))

#plot_digits(digits.data)

type(digits.data)        
dfDigits = pd.DataFrame(digits.data)
dfDigits.shape
dfDigits2 = dfDigits.to_numpy()

# model = PCA(0.5)
# components = model.fit_transform(digits.data)
# data_rec = model.inverse_transform(components)

implemented_ndimensionsmethods = ["KaiserGuttman"]

for i, method in enumerate(implemented_ndimensionsmethods):
    print(method)
    
    PCA_Actual = perform_projections(data = dfDigits, scale = True, ndimensionsmethod=method)
    
    plot_digits(dfDigits2)
    
    digitsData_PCA_Actual_PCAimp = dfDigits.copy()
    digitsData_PCA_Actual_PCAimp.shape
    PCA_Actual_PCAimp_vars = list(PCA_Actual["relevant_features_pca"].index)
    for i in range(0,digitsData_PCA_Actual_PCAimp.shape[1]):
        if i not in PCA_Actual_PCAimp_vars:
            digitsData_PCA_Actual_PCAimp[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_PCAimp.to_numpy())
    
    digitsData_PCA_Actual_PCAimpN = dfDigits.copy()
    digitsData_PCA_Actual_PCAimpN.shape
    PCA_Actual_PCAimp_varsN = list(PCA_Actual["relevant_features_pca"].index)
    for i in range(0,digitsData_PCA_Actual_PCAimpN.shape[1]):
        if i in PCA_Actual_PCAimp_varsN:
            digitsData_PCA_Actual_PCAimpN[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_PCAimpN.to_numpy())

    digitsData_PCA_Actual_PCAimpL = dfDigits.copy()
    digitsData_PCA_Actual_PCAimpL.shape
    PCA_Actual_PCAimp_varsL = list(PCA_Actual["least_relevant_features_pca"].index)
    for i in range(0,digitsData_PCA_Actual_PCAimpL.shape[1]):
        if i not in PCA_Actual_PCAimp_varsL:
            digitsData_PCA_Actual_PCAimpL[i].values[:] = 0
    plot_digits(digitsData_PCA_Actual_PCAimpL.to_numpy())


plot_digits4(dfDigits2, 
             digitsData_PCA_Actual_PCAimp.to_numpy(), 
             digitsData_PCA_Actual_PCAimpN.to_numpy(), 
             digitsData_PCA_Actual_PCAimpL.to_numpy())

