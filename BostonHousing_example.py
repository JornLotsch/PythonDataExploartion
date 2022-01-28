#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:23:50 2022

@author: joern
"""

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso#Loading the dataset
x = load_boston()
dfBoston = pd.DataFrame(x.data, columns = x.feature_names)
dfBoston["MEDV"] = x.target
X_Boston = dfBoston.drop("MEDV",1)   #Feature Matrix
y_Boston = dfBoston["MEDV"]          #Target Variable
X_Boston.head()

perform_projections(data = X_Boston, scale = True, ndimensionsmethod="ExplainedVariance")
