#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:22:35 2022

@author: joern
"""
#from scipy.interpolate import interp1d
from scipy import interpolate
#from scipy.interpolate import splev, splrep
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ABC_clean_data(data):

    # Create pandas dataframe with variable names and input values
    if not is_numeric_dtype(data):
        raise Warning("Data is not numeric")
        return

    data = data.dropna(axis=0)
    data = data[~np.isinf(data)]
    data = data.clip(lower=0)
    if isinstance(data, pd.Series):
        varnames = data.index
        data = data.tolist()
    else:
        varnames = ["Var"+str(i) for i in list(range(1, len(data)+1))]

    dfItems = pd.DataFrame(data={"varname": varnames, "value": data})
    # dfItems = pd.DataFrame() data.to_frame(name="Value")

    if len(dfItems) != len(data):
        print(str(len(dfItems)) + "rows of " + str(len(data)) +
              "items are positive and beeing used for further calculations.")

    return(dfItems)


def ABC_curve(CleanedData):

    # Create pandas dataframe with fitted ABC curve and local slopes

    CleanedData.sort_values(by="value", ascending=False, inplace=True)

    Contrib = CleanedData["value"].array
    y = np.cumsum(Contrib)
    y = y/y[-1]
    y = y[~np.isinf(y)]
    x = np.arange(1, len(y)+1)/len(y)
    if np.min(y) > 0:
        y = np.insert(y, 0, 0, axis=0)
        x = np.insert(x, 0, 0, axis=0)
    if np.max(y) < 1:
        y = np.append(y, 1)
        x = np.append(x, 1)

    f = interpolate.splrep(x, y)
    Effort = np.linspace(0, 1, num=100, endpoint=True)
    Yield = interpolate.splev(Effort, f)
    dABC = interpolate.splev(Effort, f, der=1)

    return pd.DataFrame({"effort": Effort, "yield": Yield, "dABC": dABC}, columns=["effort", "yield", "dABC"])


def ABC_calc(CleanedData, ABCcurveData):

    # Calculate set limits
    curve = ABCcurveData[["effort", "yield"]]

    distPareto = np.zeros((curve.shape[0], 1))
    point = [0.0, 1.0]
    for i in range(len(curve.iloc[:, 0])):
        pointdist = abs(point - curve.iloc[[i]])**2
        distPareto[i] = np.sum(pointdist.values.tolist())
    ParetoPointInd = np.where(distPareto == distPareto.min())[0].tolist()[0]
    ParetoPoint = curve.iloc[[ParetoPointInd]]
    ableitung = ABCcurveData["dABC"]
    ableitung = abs(ABCcurveData["dABC"] - 1)
    breakEvenInd = np.where(ableitung == ableitung.min())[0].tolist()[0]
    breakEvenPoint = curve.iloc[[breakEvenInd]]

    if curve["effort"][breakEvenInd] < curve["effort"][ParetoPointInd]:
        ABexchanged = True
        JurenInd = breakEvenInd
        Bx = curve["effort"][ParetoPointInd]
        A = breakEvenPoint
        B = ParetoPoint
    else:
        ABexchanged = False
        JurenInd = ParetoPointInd
        Bx = curve["effort"][breakEvenInd]
        A = ParetoPoint
        B = breakEvenPoint

    distBx = np.zeros((curve.shape[0], 1))
    Juren = [Bx, 1.0]
    for i in range(len(curve.iloc[:, 0])):
        Jurendist = abs(Juren - curve.iloc[[i]])**2
        distBx[i] = np.sum(Jurendist.values.tolist())

    B_limit = np.where(distBx == distBx.min())[0].tolist()[0]
    C = curve.iloc[[B_limit]]

    f = interpolate.splrep(np.linspace(1, 100, num=len(
        CleanedData["value"]), endpoint=True), CleanedData["value"])
    interpolatedInverseEcdf = interpolate.splev(
        np.linspace(1, 100, num=1000, endpoint=True), f)
    ABlimit = interpolatedInverseEcdf[round(A.values.tolist()[0][0] * 1000) + 1]
    BClimit = interpolatedInverseEcdf[round(C.values.tolist()[0][0] * 1000) + 1]

    Aind = CleanedData.loc[CleanedData['value'] > ABlimit]
    Bind = CleanedData.loc[CleanedData['value'].between(BClimit, ABlimit)]
    Cind = CleanedData.loc[CleanedData['value'] < BClimit]

    smallestAData = curve["yield"][JurenInd]
    smallestBData = curve["yield"][B_limit]

    return {"Aind": Aind, "Bind": Bind, "Cind": Cind, "ABexchanged": ABexchanged,
            "A": A, "B": B, "C": C, "smallestAData": smallestAData,
            "smallestBData": smallestBData, "AlimitIndInInterpolation": JurenInd,
            "BlimitIndInInterpolation": B_limit, "p": curve["effort"], "ABC": curve["yield"],
            "ABlimit": ABlimit, "BClimit": BClimit}

def ABC_plot(ABCresults, CleanedData):
    Contrib = CleanedData["value"].array
    y = np.cumsum(Contrib)
    y = y/y[-1]
    y = y[~np.isinf(y)]
    x = np.arange(1, len(y)+1)/len(y)
    pUnif = np.linspace(0, 1, 100)
    A = CleanedData["value"].min()
    MaxX = CleanedData["value"].max()
    if A == MaxX:
        A = 0
        MaxX = 1
    B = MaxX - A
    ABCuniform = (-0.5 * B * pUnif**2 + MaxX * pUnif)/(A + 0.5 * B)
    
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize = (10,10))
        ax = sns.scatterplot(x = x, y = y, color = "none", edgecolor="blue")
        ax.margins(x=0, y = 0)
        sns.lineplot(x = ABCresults["p"], y = ABCresults["ABC"], color = "dodgerblue")
        sns.lineplot(x=[ABCresults["A"].values.tolist()[0][0], ABCresults["A"].values.tolist()[0][0]], 
                      y = [0, ABCresults["A"].values.tolist()[0][1]], color = "salmon", linewidth  = 2)
        sns.lineplot(x=[0,ABCresults["A"].values.tolist()[0][0]], 
                      y = [ABCresults["A"].values.tolist()[0][1], ABCresults["A"].values.tolist()[0][1]], color = "salmon", linewidth = 2)
        sns.lineplot(x=[ABCresults["C"].values.tolist()[0][0], ABCresults["C"].values.tolist()[0][0]], 
                      y = [0, ABCresults["C"].values.tolist()[0][1]], color = "salmon", linewidth  = 2)
        sns.lineplot(x=[0,ABCresults["C"].values.tolist()[0][0]], 
                      y = [ABCresults["C"].values.tolist()[0][1], ABCresults["C"].values.tolist()[0][1]], color = "salmon", linewidth = 2)
        sns.lineplot(x=pUnif, y=pUnif, color = "magenta", linestyle="dashed")
        sns.lineplot(x=pUnif, y=ABCuniform, color = "green", linestyle="dotted")
        plt.text(0.5 * ABCresults["A"].values.tolist()[0][0], .1, 
                  "Set A:\nn = " + str(len(ABCresults["Aind"])), 
                  horizontalalignment='center', size='large', color='blue', weight='bold')
        plt.text(0.5 * (ABCresults["C"].values.tolist()[0][0] + ABCresults["A"].values.tolist()[0][0]), .1, 
                  "Set B:\nn = " + str(len(ABCresults["Bind"])), 
                  horizontalalignment='center', size='medium', weight='semibold')
        plt.text(0.5 * (1 + ABCresults["C"].values.tolist()[0][0]), .1, 
                  "Set C:\nn = " + str(len(ABCresults["Cind"])), 
                  horizontalalignment='center', size='medium', weight='semibold')
    return(fig)

#, Effort, Yield, '-', Effort, dABC, '--')

def ABC_analysis(data, PlotIt=False):
    CleanedData = ABC_clean_data(data)
    ABCcurveData = ABC_curve(CleanedData = CleanedData)
    ABCresults = ABC_calc(CleanedData = CleanedData, ABCcurveData = ABCcurveData)
    if PlotIt:
        ABC_plot(ABCresults = ABCresults, CleanedData = CleanedData)
    return ABCresults    
    
