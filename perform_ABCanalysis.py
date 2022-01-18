#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:22:35 2022

@author: joern
"""
#from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import splev, splrep
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ABCcleanData(data):
    
    if not is_numeric_dtype(data):
        raise Warning("Data is not numeric")
        return

    data = data.dropna(axis=0)
    data =  data[~np.isinf(data)]
    data = data.clip(lower=0)
    dfItems = data.to_frame(name="Value")
     
    if len(dfItems) != len(data):
        print(str(len(dfItems)) + "rows of " + str(len(data)) + "items are positive and beeing used for further calculations.")
    
    return(dfItems)
   
xx = ABCcleanData(data)
type(xx) 

df_data = xx

def ABCcurve(df_data):
    
    df_data.sort_values(by = "Value", ascending=False, inplace=True)

    Contrib = df_data["Value"].array
    y = np.cumsum(Contrib)
    y = y/y[-1]
    y =  y[~np.isinf(y)]
    x = np.arange(1,len(y)+1)/len(y)
    if np.min(y) > 0:
        y = np.insert(y, 0, 0, axis = 0)
        x = np.insert(x, 0, 0, axis = 0)
    if np.max(y) < 1:
        y = np.append(y, 1)
        x = np.append(x, 1)
    
    f = interpolate.splrep(x, y)
    xnew = np.linspace(0, 1, num=1000, endpoint=True)
    ynew = interpolate.splev(xnew, f) 
    ynew_der = interpolate.splev(xnew, f, der=1) 
    Effort = xnew
    Yield = ynew
    dABC = ynew_der
    
    type(Effort)
    
    plt.plot(x, y, 'o', xnew, ynew, '-', xnew, ynew_der, '--')
    plt.show()
    
    return {"Curve": pd.DataFrame({'Effort': Effort, 'Yield': Yield}, columns=['Effort', 'Yield']), 
            "Slope": pd.DataFrame({'xnew': xnew, 'dABC': dABC}, columns=['xnew', 'dABC'])}

yy = ABCcurve(xx)


def ABCanalysis(data, ABCcurvedata, PlotIt = False):

    plt.plot(curve["Effort"], curve["Yield"], 'o')
    plt.show()
    
    curve = yy["Curve"]
    #curve = pd.DataFrame([[1,2,3,5,4,3], [2,3,4,6,5,4]]).T
    distPareto = np.zeros((curve.shape[0],1))
    point = [0.0, 1.0]
    for i in range(len(curve.iloc[:,0])): 
        pointdist = abs(point - curve.iloc[[i]])**2
        distPareto[i] = np.sum(pointdist.values.tolist())
 
    
aa =  np.where(distPareto == distPareto.min())[0].tolist()[0]



    result = np.where(distPareto == np.amin(distPareto))
listOfCordinates = list(zip(result[0], result[1]))
listOfCordinates[0]

    np.amin(distPareto)
    
    plt.plot(curve["Effort"], '-')
    plt.show()     
        
        {
  distPareto[i] = sum(abs(point - curve[i, ])^2)
}

distPareto[2]

#     xnew = np.arange(0,1,0.001)
#     df_data.sort_values(by = "Value", ascending=False, inplace=True)
#     Contrib = df_data["Value"].array
#     y = np.cumsum(Contrib)
#     y = y/y[-1]
#     y =  y[~np.isinf(y)]
#     x = np.arange(1,len(y)+1)/len(y)
#     if np.min(y) > 0:
#         y = np.insert(y, 0, 0, axis = 0)
#         x = np.insert(x, 0, 0, axis = 0)
#     if np.max(y) < 1:
#         y = np.append(y, 1)
#         x = np.append(x, 1)
    
    
#     f2 = interpolate.splrep(x, y) # Added

#     minimo = min(x)
#     maximo = max(y)

# xnew = np.linspace(minimo, maximo, num=400, endpoint=True)
# ynew = interpolate.splev(xnew, f2) # Added 
# y
    
#     f = interp1d(x, y)
#     ynew = f(xnew) 

#     f2 = interpolate.splrep(x_ordenado, y_ordenado_simetric) # Added

#     ynew2 = interpolate.splev(xnew, f2) # Added 
#     ynew_der = interpolate.splev(xnew, f2, der=1) # Added to compute first derivative

    




    

# {
#     if (!is.vector(Data)) {
#         n = nrow(Data)
#         d = ncol(Data)
#         warning("Only vectors should be used!")
#         if (d > 1) {
#             warning("Using only first column of data")
#             UncleanData = as.vector(Data[, 1])
#         }
#         else {
#             UncleanData = Data
#         }
#     }
#     else {
#         UncleanData = Data
#     }
#     UncleanData = as.numeric(unname(UncleanData))
#     rowsbefore = length(UncleanData)
#     nabools = is.finite(UncleanData)
#     Data2CleanInd = which(nabools == FALSE)
#     CleanData = UncleanData
#     if (length(Data2CleanInd)) 
#         CleanData[Data2CleanInd] = 0
#     DataNeg = CleanData[CleanData < 0]
#     cols = 1
#     bools = CleanData %in% DataNeg
#     CleanData[bools] <- 0
#     rows = rowsbefore - sum(bools) - sum(!nabools)
#     if (rowsbefore > rows) {
#         warning(paste0(rows, " of ", rowsbefore, " items are positive and beeing used for further calculations."))
#     }
#     return(list(CleanedData = CleanData, Data2CleanInd = Data2CleanInd))
# }


# ABCcurve
# function (Data, p) 
# {
#     cleanData = ABCcleanData(Data)$CleanedData
#     rows = length(cleanData)
#     if (missing(p)) {
#         if (rows < 101) {
#             p = seq(from = 0, to = 1, by = 0.01)
#         }
#         else {
#             p = seq(from = 0, to = 1, by = 0.001)
#         }
#     }
#     sorted = sort(na.last = T, cleanData, decreasing = TRUE)
#     Anteil = sorted
#     y = cumsum(Anteil)
#     y = y/tail(y, 1)
#     y[is.nan(y)] = 0
#     x = (1:rows)/rows
#     if (head(y, 1) > 0) {
#         x = c(0, x)
#         y = c(0, y)
#     }
#     if (tail(x, 1) < 1) {
#         x = c(x, 1)
#         y = c(y, 1)
#     }
#     V = spline(x, y, xout = p)
#     Effort = V$x
#     Yield = V$y
#     inds = which(Yield >= 1)
#     ind1 = min(inds)
#     if (ind1 < length(Yield)) 
#         Yield[c(ind1:length(Yield))] = 1
#     n = length(Effort)
#     Curvengleichung = splinefun(Effort, Yield)
#     ableitung = Curvengleichung(1:n/n, 1)
#     return(list(Curve = cbind(Effort = Effort, Yield = Yield), 
#         CleanedData = cleanData, Slope = cbind(p = p, dABC = ableitung)))
# }


ABCanalysis
function (Data, ABCcurvedata, PlotIt = FALSE) 
{
    requireNamespace("plotrix")
    if (missing(Data)) {
        if (missing(ABCcurvedata)) {
            stop("argument \"Data\" and ABCcurvedata are missing")
        }
        else {
            Data = NULL
        }
    }
    if (!PlotIt) {
        if (missing(ABCcurvedata)) {
            ABCcurvedata = ABCcurve(Data)
        }
        Effort = ABCcurvedata$Curve[, "Effort"]
        Yield = ABCcurvedata$Curve[, "Yield"]
        curve = cbind(Effort, Yield)
        distPareto = c()
        point = t(as.matrix(c(0, 1)))
        for (i in 1:length(Effort)) {
            distPareto[i] = sum(abs(point - curve[i, ])^2)
        }
        ParetoPointInd = which.min(distPareto)
        ParetoPoint = curve[ParetoPointInd, ]
        ableitung = ABCcurvedata$Slope[, "dABC"]
        BreakEvenInds = which.min(abs(ableitung - 1))
        BreakEvenInd = max(BreakEvenInds)
        BreakEvenPoint = curve[BreakEvenInd, ]
        if (Effort[BreakEvenInd] < Effort[ParetoPointInd]) {
            ABexchanged = TRUE
            JurenInd = BreakEvenInd
            Bx = Effort[ParetoPointInd]
            A = BreakEvenPoint
            B = ParetoPoint
        }
        else {
            JurenInd = ParetoPointInd
            Bx = Effort[BreakEvenInd]
            ABexchanged = FALSE
            A = ParetoPoint
            B = BreakEvenPoint
        }
        distBx = c()
        Juren = t(as.matrix(c(Bx, 1)))
        for (i in 1:length(Effort)) {
            distBx[i] = sum(abs(Juren - curve[i, ])^2)
        }
        bgrenze = which.min(distBx)
        C = curve[bgrenze[1], ]
        if (!is.null(Data)) {
            sortedData = sort(Data, decreasing = T)
            interpolatedInverseEcdf = spline(seq(1, 100, length.out = length(Data)), 
                sortedData, n = 1000)
            ABLimit = interpolatedInverseEcdf$y[round(A[1] * 
                1000) + 1]
            BCLimit = interpolatedInverseEcdf$y[round(C[1] * 
                1000) + 1]
            Aind = which(Data > ABLimit)
            Bind = which((Data <= ABLimit) & (Data >= BCLimit))
            Cind = which(Data < BCLimit)
        }
        else {
            Bind = NULL
            Cind = NULL
            Aind = NULL
            warning("No Data given: Calculating curve and points by given ABCcurvedata")
        }
        return(list(Aind = Aind, Bind = Bind, Cind = Cind, ABexchanged = ABexchanged, 
            A = A, B = B, C = C, smallestAData = Yield[JurenInd], 
            smallestBData = Yield[bgrenze], AlimitIndInInterpolation = JurenInd, 
            BlimitIndInInterpolation = bgrenze, p = Effort, ABC = Yield, 
            ABLimit = ABLimit, BCLimit = BCLimit))
    }
    else {
        if (missing(Data) | is.null(Data)) {
            abc = ABCanalysisPlot(ABCcurvedata = ABCcurvedata)$ABCanalysis
        }
        else {
            abc = ABCanalysisPlot(Data)$ABCanalysis
        }
    }
}