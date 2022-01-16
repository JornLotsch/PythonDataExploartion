#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:44:58 2022

@author: joern
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats


def perform_pca(data, y, scale = True, showonlyPC12 = False):
    
    if y.ndim == 1:
        y1 = y.iloc[:,0]
        y2 = pd.DataFrame([0] * len(y1))
    else:
        y1 = y.iloc[:,0]
        y2 = y.iloc[:,1]
    
    y_comb = []
    for i in range(len(y1)):
        y_comb.append(str(y1[i])+str(y2[i]))
        
    X = data
    
    if scale == True:
        X_sc = StandardScaler().fit_transform(X)
    else:
        X_sc = X
    
    model_PCA = PCA()
    principalComponents = model_PCA.fit_transform(X_sc)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, y1, y2], axis=1)
    
    loadings = model_PCA.components_
    num_pc = model_PCA.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = X.columns.values
    loadings_df = loadings_df.set_index('variable')
    loadings_df

    cumvar = np.cumsum(model_PCA.explained_variance_ratio_)
    indexGE90 = min([index for index in range(
        len(cumvar)) if cumvar[index] >= 0.9])
    explainedvar = model_PCA.explained_variance_
    nPCs = len(explainedvar[explainedvar>= 1])
    if nPCs < 2:
        nPCs = 2
    
    stats_p_y1 = []
    stats_p_y2 = []
    for i in range(0,nPCs + 1):
        if len(np.unique(y1)) > 1:
            stats_p_y1.append(stats.mannwhitneyu(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:,-2])]).pvalue)
        if len(np.unique(y2)) > 1:
            stats_p_y2.append(stats.mannwhitneyu(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:,-1])]).pvalue)
    
    significantPC_y1 = [i for i in range(len(stats_p_y1)) if stats_p_y1[i] < 0.05]
    significantPC_y2 = [i for i in range(len(stats_p_y2)) if stats_p_y2[i] < 0.05]
    
    showPCs = [0, 1]
    if not showonlyPC12:
        showPCs = np.unique(np.sort(significantPC_y1 + significantPC_y2))
        if len(showPCs) < 1:
            showPCs = [0, 1]
        elif (len(showPCs) < 2):
             showPCs = [showPCs[0], 1]
        else: showPCs = showPCs[:2]
    
    model_PCA_reconstruction = PCA(n_components=nPCs)
    principalComponents_reconstruction = model_PCA_reconstruction.fit_transform(X_sc)
    X_reconstructed = model_PCA_reconstruction.inverse_transform(principalComponents_reconstruction)
    mean_sq_reconstructionError = 1/np.sqrt(((X_reconstructed - X) ** 2).mean())
    mean_sq_reconstructionError.sort_values(axis=0, ascending=False, inplace=True)
    
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(23, 16))
        gs0 = plt.GridSpec(9, 6, figure=fig, wspace=.8, hspace=.8)
    
        ax1 = fig.add_subplot(gs0[1:7, :3])
        ax2 = fig.add_subplot(gs0[0, :3])
        ax3 = fig.add_subplot(gs0[1:7, 3])
        ax4 = fig.add_subplot(gs0[0:3, 4:])
        ax5 = fig.add_subplot(gs0[3:6, 4:])
        ax6 = fig.add_subplot(gs0[7:, :4])
        ax7 = fig.add_subplot(gs0[6:, 4:])
    
        sns.scatterplot(
            ax=ax1, x=finalDf[showPCs[0]], y=finalDf[showPCs[1]],  hue=finalDf.iloc[:,-2], style=finalDf.iloc[:,-1], cmap="viridis")
        ax1.set_xlabel(pc_list[showPCs[0]])
        ax1.set_ylabel(pc_list[showPCs[1]])
        ax1.set_title("PCA of scaled variables")
    
        xvector = model_PCA.components_[0]
        yvector = model_PCA.components_[1]
    
        for i in range(len(xvector)):
            ax1.arrow(x=0, y=0, dx=xvector[i]*max(finalDf[showPCs[0]]), dy=yvector[i]*max(finalDf[showPCs[1]]),
                      color="salmon", width=0.005, head_width=0.05)
            ax1.text(xvector[i]*max(finalDf[showPCs[0]])*1.1, yvector[i]*max(finalDf[showPCs[1]])*1.1,
                     list(X.columns.values)[i], color="salmon")
    
        sns.kdeplot(ax=ax2, x=finalDf[showPCs[0]], hue=y_comb)
        ax2.text(min(finalDf[showPCs[0]]), 0.8 * ax2.get_ylim()[1], "U test " + y.columns[0] + "p = " +
                  "{:.2e}".format(stats_p_y1[showPCs[0]]), va="baseline")
        ax2.text(min(finalDf[showPCs[0]]), 0.4 * ax2.get_ylim()[1], "U test " + y.columns[1] + "p = " +
                  "{:.2e}".format(stats_p_y1[showPCs[1]]), va="baseline")
        sns.kdeplot(ax=ax3, y=finalDf[showPCs[1]], hue=y_comb)
        ax3.text(0.8 * ax3.get_xlim()[1], np.mean(finalDf[showPCs[1]]), "U test " + y.columns[0] + "p = " + "{:.2e}".format(stats_p_y2[showPCs[0]]),
                  rotation=-90, va="top")
        ax3.text(0.4 * ax3.get_xlim()[1], np.mean(finalDf[showPCs[1]]), "U test " + y.columns[1] + "p = " + "{:.2e}".format(stats_p_y1[showPCs[1]]),
                  rotation=-90, va="top")
    
        sns.lineplot(ax=ax4, x=range(1, len(model_PCA.components_)+1),
                     y=np.cumsum(model_PCA.explained_variance_ratio_))
        ax4.set_xlabel("PC count")
        ax4.set_ylabel("Cumulative explaned variance")
        ax4.set_title("PCA component explained variance")
        ax4.axhline(0.9, color="salmon")
        ax4.text(indexGE90, 0.85, "90 % variance expianed with " + str(indexGE90) + " PCs",
                  rotation=90, va="top", color="red")
    
        sns.barplot(ax=ax5, x=pc_list, y = explainedvar, color="dodgerblue")
        ax5.set_xlabel("PC")
        ax5.set_ylabel("Eigenvalue")
        ax5.set_title("Eigenvalues")
        ax5.axhline(1, color = "black")
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)

        PCAcomponents = model_PCA.components_[0:nPCs]
        sns.heatmap(ax=ax6, data=PCAcomponents,
                     cmap="BrBG",
                     yticklabels=["PCA"+str(x) for x in range(1, nPCs+1)],
                     xticklabels=list(X.columns),
                     cbar_kws={"orientation": None})
    
        sns.barplot(ax=ax7, x=mean_sq_reconstructionError.index,
                    y=mean_sq_reconstructionError, color="dodgerblue")
        ax7.set_xlabel("Variables")
        ax7.set_ylabel("1/RMSE")
        ax7.set_title("PCA reconstruction RMSE")
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=90)
        ax7.text(len(X.columns)/2, 0.95 * np.max(mean_sq_reconstructionError), "Overal mean RMSE: " +
                  str(round(np.mean(mean_sq_reconstructionError), 3)), va="baseline", color="red")

