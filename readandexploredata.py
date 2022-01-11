#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""

import pandas as pd
import numpy as np

pfad_o = "/home/joern/Aktuell/RiechenVerduennungAbstand/"
pfad_u1 = "09Originale/"
filename = "data_anne_huster_dec_2021.xlsx"
sheetname = "data"

dfRiechenVerduennungAbstand = pd.read_excel(pfad_o + pfad_u1 + filename, sheet_name = sheetname)
dfRiechenVerduennungAbstand.columns
dfRiechenVerduennungAbstand.index 
xx = dfRiechenVerduennungAbstand.set_index(["Sample ID","sex_0f", "YOB"])
mxx = xx["BMI"].mean(level = ["sex_0f", "YOB"])
xx.index.is_unique
print(xx.index)

import datetime
Now = datetime.date.today()
dfRiechenVerduennungAbstand["AgeNow"] = Now.year - dfRiechenVerduennungAbstand["YOB"]

id_list = list(dict.fromkeys(dfRiechenVerduennungAbstand.index))
dfRiechenVerduennungAbstand.index.symmetric_difference(id_list)

missingsDetected = dfRiechenVerduennungAbstand.isnull().any(axis=0)
missingsDetected[missingsDetected == True]

for i in range(len(dfRiechenVerduennungAbstand["YOB"])):
    if dfRiechenVerduennungAbstand["YOB"][i] > 1900:
        year = dfRiechenVerduennungAbstand["YOB"][i]
    else:
        year = 2000
    print(pd.to_datetime({"year": [year], "month": [1], "day": [1]}))

import copy
years = copy.copy(dfRiechenVerduennungAbstand["YOB"])
years2 = dfRiechenVerduennungAbstand["YOB"] 
dfRiechenVerduennungAbstand["YOB"].plot(kind='hist', density = False)
years2.plot(kind = "hist")

years2[years2 < 1900] = 1950

df=pd.DataFrame({"year": list(years), 
                "month": list(years*0+1), 
                "day": list(years*0+1)})

Dates = pd.to_datetime(df)
dfRiechenVerduennungAbstand.YOB2 = list(pd.DatetimeIndex(Dates).year)
np.mean(dfRiechenVerduennungAbstand.YOB2 )
import seaborn as sns
sns.histplot(data = dfRiechenVerduennungAbstand, x = "YOB")



np.mean(dfRiechenVerduennungAbstand.BMI)
sum(dfRiechenVerduennungAbstand.BMI)
dfRiechenVerduennungAbstand.BMI.mean()
sum(dfRiechenVerduennungAbstand.BMI.isnull()==True)
sum(dfRiechenVerduennungAbstand.BMI.notnull()==True)
np.mean(dfRiechenVerduennungAbstand.BMI[dfRiechenVerduennungAbstand.BMI.notnull()])
np.mean(dfRiechenVerduennungAbstand.BMI[pd.notnull(dfRiechenVerduennungAbstand.BMI)])

dfRiechenVerduennungAbstand.iloc[:,1:5].dropna().describe()
dfRiechenVerduennungAbstand.iloc[:,1:5].dropna().std()



import numpy as np
print("Mittlerer BMI: ", np.nanmean(dfRiechenVerduennungAbstand.BMI))

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

plt.scatter(dfRiechenVerduennungAbstand.YOB2, dfRiechenVerduennungAbstand.BMI)


plt.figure(0)
plt.hist(dfRiechenVerduennungAbstand.BMI.plot, 40, histtype = "step")


for i in range(len(dfRiechenVerduennungAbstand.columns)):
    variable2plot = dfRiechenVerduennungAbstand[dfRiechenVerduennungAbstand.columns[i]]
    if np.issubdtype(variable2plot.dtype, np.number):
        if sum(pd.notna(variable2plot)) > 3:
            variable2plot = variable2plot[pd.notna(variable2plot)]
            plt.figure(i)
            if variable2plot.std() > 0:
                variable2plot.plot(kind='hist', density = True)
                #variable2plot.plot(kind='density')
                variable2plot.plot.kde(zorder=2, color='blue')
            else:
                variable2plot.plot(kind='hist', density = True)
            plt.title(dfRiechenVerduennungAbstand.columns[i])

                

plt.title("BMIs")
plt.xlabel("BMI")
plt.ylabel("Count")

plt.figure(1)
plt.hist(dfRiechenVerduennungAbstand.BMI[dfRiechenVerduennungAbstand.BMI > 20], 40)

plt.figure(2)
plt.scatter(x = dfRiechenVerduennungAbstand.BMI, y = dfRiechenVerduennungAbstand.BMI)
plt.title("BMIs")
plt.xlabel("BMI")
plt.ylabel("BMI")

plt.scatter(x = dfRiechenVerduennungAbstand.BMI, y = dfRiechenVerduennungAbstand.BMI, alpha = 0.1)
np.random.seed(42)
indices = np.random.choice(1000, size = len(dfRiechenVerduennungAbstand.BMI), replace = True)
min_indices = 800
plt.scatter(x = dfRiechenVerduennungAbstand.BMI[indices > min_indices ], 
            y = dfRiechenVerduennungAbstand.BMI[indices > min_indices ], 
            color = "red", facecolor = "none", s = 200)
np.random.seed(421)
indices = np.random.choice(1000, size = len(dfRiechenVerduennungAbstand.BMI), replace = True)
min_indices = 800
plt.scatter(x = dfRiechenVerduennungAbstand.BMI[indices > min_indices ], 
            y = dfRiechenVerduennungAbstand.BMI[indices > min_indices ], 
            color = "blue", facecolor = "none", s = 200)


bins = np.linspace(np.min(dfRiechenVerduennungAbstand.BMI), 
                   np.max(dfRiechenVerduennungAbstand.BMI),  40)
count = np.zeros_like(bins)
n_per_bin = np.searchsorted(bins, dfRiechenVerduennungAbstand.BMI)
np.add.at(count, n_per_bin,1)

plt.figure(4)
plt.step(bins, count)
plt.title("BMIs")
plt.xlabel("BMI")
plt.ylabel("Count")

fig7 = seaborn.violinplot("YOB", "BMI", data = dfRiechenVerduennungAbstand, hue = "sex_0f")
#fig7.set_xticklabels(fig7.get_xticklabels(), rotation=90)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small' )

# Linear regression

df = dfRiechenVerduennungAbstand[["AgeNow", "BMI"]] 
df = df[df["AgeNow"] < 100]
df = df.dropna()
df.columns
x = df["AgeNow"]
y = df["BMI"]

plt.plot(x)

x.shape
y.shape
X = x[:,np.newaxis]
Y = y[:,np.newaxis]
X.shape
Y.shape

plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

model = LinearRegression(fit_intercept = True)
model

est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())
      
linreg = model.fit(X, y)
linreg.intercept_
linreg.coef_

xfit = np.linspace(x.min(), x.max())
Xfit = xfit[:,np.newaxis]

yfit = linreg.predict(Xfit)
plt.scatter(x, y, color = "red")
plt.plot(xfit, yfit)

residuals = (np.concatenate(Y) - linreg.predict(Y)) / linreg.predict(Y)
residuals = (np.concatenate(Y) - linreg.predict(Y)) 

#plt.plot(residuals)

fig8 = plt.figure()
grid = plt.GridSpec(2, 3, wspace = 0.4, hspace = 0.4)
plt.subplot(grid[0,:3]).scatter(x, y, color = "green", marker = "o")
plt.subplot(grid[0,:3]).plot(xfit, yfit, color = "darkgreen")
plt.subplot(grid[1,:2]).plot(residuals, color = "green")
plt.subplot(grid[1,2]).hist(residuals, orientation = "horizontal", 
                bins=np.linspace(min(residuals), max(residuals), 20),
                color = "chartreuse", edgecolor = "green")

x_plot = np.arange(len(residuals))

fig9 = plt.figure()
grid = plt.GridSpec(2, 3, wspace = 0.4, hspace = 0.4)
plt.subplot(grid[:,:2]).scatter(x, y, color = "green", marker = "o")
plt.subplot(grid[:,:2]).plot(xfit, yfit, color = "darkgreen")
plt.subplot(grid[0,2]).plot(residuals, x_plot, color = "green")
plt.subplot(grid[1,2]).hist(residuals, orientation = "vertical", 
                bins=np.linspace(min(residuals), max(residuals), 20),
                color = "chartreuse", edgecolor = "green")

# Ueberwactes Lernen iris
import seaborn as sns

iris = sns.load_dataset("iris")
iris.head()
X_iris = iris.iloc[:,0:4]
Y_iris = iris.iloc[:,4]

from sklearn.model_selection import train_test_split
np.random.seed(42)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, Y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

np.random.seed(42)
model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)

from sklearn.metrics import balanced_accuracy_score
bacc = balanced_accuracy_score(ytest, y_model)
print(bacc)
from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest, y_model)
print(acc)

#PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(X_iris)
X_2D = model.transform(X_iris)

iris["PC1"] = X_2D[:,0]
iris["PC2"] = X_2D[:,1]

plt.scatter(iris["PC1"], iris["PC2"],    c = iris.iloc[:,4].replace(["setosa", "versicolor", "virginica"],
                        [1,2,3], inplace=False), cmap = "viridis", alpha = 0.4)

sns.lmplot("PC1", "PC2", data = iris, hue = "species", fit_reg=False)

# GMM clustern
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, covariance_type="full")
model.fit(X_iris)
y_fit = model.predict(X_iris)
iris["cluster"] = y_fit

sns.lmplot("PC1", "PC2", data = iris, hue = "species", col = "cluster", fit_reg=False)
