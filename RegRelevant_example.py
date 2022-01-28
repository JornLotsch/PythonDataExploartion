#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:13:08 2022

@author: joern
"""

# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/

import pandas as pd
import numpy as np
import seaborn as sns

# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# generate dataset
X, y = make_regression(n_samples=100, n_features=100, n_informative=10, random_state = 42 )
# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
dfX = pd.DataFrame(X)
dfX.columns = ["col" + str(i) for i in (range(len(dfX.columns)))]

X_selected = fs.fit_transform(X, y)
print(X_selected.shape)

dfX_selected = pd.DataFrame(X_selected)

equal = []
for i1 in range(len(dfX.columns)):
    for i2 in range(len(dfX_selected.columns)):
        if np.sum(dfX.iloc[:,i1] - dfX_selected.iloc[:,i2]) == 0:
            equal.append(i1)

print(equal)

perform_projections(data = dfX, scale = False, ndimensionsmethod="KaiserGuttman")

sns.violinplot(data = dfX)