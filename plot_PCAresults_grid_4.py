#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 17:48:07 2022

@author: joern
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np


def create_projectionplot(data, scale, biplot, showPCs, model_projection, ax=None):
    ax = ax or plt.gca()
    sns.scatterplot(ax=ax, x=data.iloc[:, showPCs[0]].values, y=data.iloc[:, showPCs[1]
                                                                          ].values, hue=data.iloc[:, -2], style=data.iloc[:, -1], palette="bright")
    ax.set_xlabel("Dim " + str(showPCs[0]+1))
    ax.set_ylabel("Dim " + str(showPCs[1]+1))
    ax.set_title("PCA projection of " +
                 ("scaled variables" if scale == True else "variables"))

    if biplot == True:
        xvector = model_projection.components_[0]
        yvector = model_projection.components_[1]

        for i in range(len(xvector)):
            ax.arrow(x=0, y=0, dx=xvector[i]*max(data.iloc[:, showPCs[0]].values), dy=yvector[i]*max(data.iloc[:, showPCs[1]].values),
                     color="salmon", width=0.005, head_width=0.05)
            ax.text(xvector[i]*max(data.iloc[:, showPCs[0]].values)*1.1, yvector[i]*max(data.iloc[:, showPCs[1]].values)*1.1,
                    data.columns.values[i], color="red")
    return


def create_projectionplot_marginaldits(data, showPC, n_dims, y1, y2, y_comb, stats1, stats2, rotation=0, ax=None):
    ax = ax or plt.gca()
    if rotation == 0:
        sns.kdeplot(ax=ax, x=data, hue=y_comb)
        if len(stats1) > 0:
            ax.text(min(data), 0.8 * ax.get_ylim()[1], "Group differences " + str(y1.name) + "p = " +
                    "{:.2e}".format(stats1[showPC]), va="baseline")
        if len(stats2) > 0:
            ax.text(min(data), 0.4 * ax.get_ylim()[1], "Group differences " + str(y2.name) + "p = " +
                    "{:.2e}".format(stats2[showPC]), va="baseline")
        ax.set_xlabel(None) 
    else:
        sns.kdeplot(ax=ax, y=data, hue=y_comb)
        if n_dims > 1:
            if len(stats1) > 0:
                ax.text(0.8 * ax.get_xlim()[1], np.mean(data), "Group differences " + str(y1.name) + "p = " + "{:.2e}".format(stats1[showPC]),
                        rotation=rotation, va="top")
            if len(stats2) > 0:
                ax.text(0.4 * ax.get_xlim()[1], np.mean(data), "Group differences " + str(y2.name) + "p = " + "{:.2e}".format(stats2[showPC]),
                        rotation=-90, va="top")
        ax.set_ylabel(None)
    return


def create_lineplot(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    sns.lineplot(ax=ax, x=x, y=y, **kwargs)
    return


def create_scatterplot(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    sns.scatterplot(ax=ax, x=x, y=y, **kwargs)
    return


def create_violin_and_boxplot(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    sns.violinplot(ax=ax, data=data)
    plt.setp(ax.collections, alpha=.3)
    sns.boxplot(ax=ax, data=data, fliersize=0, **kwargs)
    return


def create_barplot(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    sns.barplot(ax=ax, x=x, y=y, **kwargs)
    return


def create_heatmap(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    sns.heatmap(ax=ax, data=data, cbar_kws={"orientation": None}, **kwargs)
    return


def plot_results_grid(ndimensionsmethod, showPCs, finalDf, scale, biplot, model_projection, X_sc, X_dist, y1, y2, y_comb, stats_p_y1, stats_p_y2,
                      reconstructionError, reconstructionError_dists, num_pc, X_reconstructed, X_reconstructed_dists,
                      num_features, indexGE90, pc_list, eigenvalues,
                      n_dims, loadings_df_z, feature_imp_mat, feature_imp,
                      relevant_variables_PCA):


    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(24, 14))
        gs0 = gridspec.GridSpec(10, 18, figure=fig, wspace=1.2, hspace=1.2)

        ax1 = fig.add_subplot(gs0[2:10, :8])  # projection
        ax2 = fig.add_subplot(gs0[:2, :8])  # marginal distribution x
        ax3 = fig.add_subplot(gs0[2:10, 8:10])  # marginal distribution y

        ax4 = fig.add_subplot(gs0[0:2, 10:])  # PCA explained variance
        ax5 = fig.add_subplot(gs0[2:4, 10:])  # PCA eigenvalues
        
        ax11 = fig.add_subplot(gs0[4:, 10:13])  # PCA loadings z heatmap
        ax12 = fig.add_subplot(gs0[4:, 13:16])  # PCA abs loadings z * egenvalues heatmap
        ax13 = fig.add_subplot(gs0[4:, 16:]) # PCA fatcor imprtance ABC barplot


        
        # ax6 = fig.add_subplot(gs0[4:6, 6:])  # reconstruction error
        # # data reconstruction vs orginal values scatterplot
        # ax7 = fig.add_subplot(gs0[6:8, 6:8])
        # # data reconstruction vs orginal values violinplot
        # ax8 = fig.add_subplot(gs0[6:10, 8:9])
        # # data reconstruction vs orginal distances violinplot
        # ax9 = fig.add_subplot(gs0[6:10, 9:])
        # # data reconstruction vs orginal distances scatterplot
        # ax10 = fig.add_subplot(gs0[8:10, 6:8])

        # ax11 = fig.add_subplot(gs0[10:, :2])  # PCA loadings z heatmap
        # # PCA abs loadings z * egenvalues heatmap
        # ax12 = fig.add_subplot(gs0[10:, 2:4])
        # # PCA fatcor imprtance ABC barplot
        # ax13 = fig.add_subplot(gs0[10:, 4:6])
        # # Reconstruction error values ABC barplot
        # ax14 = fig.add_subplot(gs0[10:, 6:8])
        # # Reconstruction error distances ABC barplot
        # ax15 = fig.add_subplot(gs0[10:, 8:])

        # number of dimenstions used in final calculations
        # ax00 = fig.add_subplot(gs0[0, 10:])
        # ax00.xaxis.set_visible(False)
        # ax00.yaxis.set_visible(False)
        # ax00.text(0.5, 0.3, "Crtierion:" + str(ndimensionsmethod), color="red",
        #           va="center", ha="center", fontsize=12, weight='light')
        # ax00.text(0.5, 0.7, "Dimensions:" + str(n_dims), color="red",
        #           va="center", ha="center", fontsize=14, weight='bold')


        create_projectionplot(ax=ax1, data=finalDf, scale=scale,
                              biplot=biplot, showPCs=showPCs, model_projection=model_projection)

        create_projectionplot_marginaldits(
            ax=ax2, data=finalDf.iloc[:, showPCs[0]], showPC=0, n_dims=n_dims, y1=y1, y2=y2, y_comb=y_comb, stats1=stats_p_y1, stats2=stats_p_y2)
        ax2.set_xlim(ax1.get_xlim())
        create_projectionplot_marginaldits(
            ax=ax3, data=finalDf.iloc[:, showPCs[1]], showPC=1, n_dims=n_dims, y1=y1, y2=y2, y_comb=y_comb, stats1=stats_p_y1, stats2=stats_p_y2, rotation=-90)
        ax3.set_ylim(ax1.get_ylim())

        create_lineplot(ax=ax4, x=range(1, len(model_projection.components_)+1),
                        y=np.cumsum(model_projection.explained_variance_ratio_), marker="o")
        ax4.set_xlabel("PC count")
        ax4.set_ylabel("Cumulative explaned variance")
        ax4.set_title("PCA component explained variance")
        ax4.axhline(0.9, color="salmon", linestyle="dotted")
        ax4.text(num_pc*0.5, 1.1 * min(np.cumsum(model_projection.explained_variance_ratio_)), 
                 "90 % variance expianed with " + str(indexGE90) + " PCs",
                 rotation=0, va="center", color="red")

        create_barplot(ax=ax5, x=pc_list, y=eigenvalues,
                       color="dodgerblue")
        ax5.set_xlabel(None)
        ax5.set_ylabel("Eigenvalue")
        ax5.set_title("Eigenvalues")
        ax5.axhline(1, color="black", linestyle="dotted")
        #ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
        ax5.text(0.5, 0.95 * np.max(eigenvalues), "Relevant dimensions (Kaiser-Guttman): " +
                 str(np.argmax(eigenvalues < 1)), va="center", color="red")
        ax5.text(0.5, 0.55 * np.max(eigenvalues), "Relevant dimensions (ABC): " +
                 str(n_dims), va="center", color="red")

        # df_reconstructionserrors = pd.DataFrame({"Values": reconstructionError.flatten().tolist(),
        #                                          "Distances": reconstructionError_dists.flatten().tolist()})
        # create_lineplot(ax=ax6, x=np.arange(
        #     1, num_pc+1), y=df_reconstructionserrors.iloc[:, 0], marker="o", label="Values")
        # create_lineplot(ax=ax6, x=np.arange(
        #     1, num_pc+1), y=df_reconstructionserrors.iloc[:, 1], marker="o", label="Distances")
        # ax6.set_xlabel("Dimensions count")
        # ax6.set_ylabel("RMSE")
        # ax6.set_title("Reconstruction error")

        # orgDataArray = X_sc.to_numpy().ravel()
        # reconstrDataArray = X_reconstructed.ravel()

        # create_scatterplot(ax=ax7, x=orgDataArray,
        #                    y=reconstrDataArray, color="dodgerblue")
        # create_lineplot(ax=ax7, x=[min(np.concatenate([orgDataArray, reconstrDataArray])),
        #                            max(np.concatenate([orgDataArray, reconstrDataArray]))],
        #                 y=[min(np.concatenate([orgDataArray, reconstrDataArray])),
        #                    max(np.concatenate([orgDataArray, reconstrDataArray]))], linestyle="dotted")
        # sns.regplot(ax=ax7, x=orgDataArray, y=reconstrDataArray,
        #             scatter=False, ci=95, fit_reg=True, color='blue')
        # ax7.set_xlabel("Original data")
        # ax7.set_ylabel("Reconstructed data")
        # ax7.set_title("Data reconstruction")
        # ax7.text(0.95 * min(np.concatenate([orgDataArray, reconstrDataArray])),
        #          0.95 *
        #          max(np.concatenate([orgDataArray, reconstrDataArray])),
        #          "Overal mean standardized RMSE: " + str(round(mean_reconstructionError_values, 3)), va="center", color="red")

        # dforig_and_reconstructed_data = pd.DataFrame(
        #     list(zip(orgDataArray, reconstrDataArray)), columns=["Orig.", "Reconstr."])
        # create_violin_and_boxplot(
        #     ax=ax8, data=dforig_and_reconstructed_data, saturation=0.5, width=0.2)
        # ax8.set_xlabel("Data sets")
        # ax8.set_ylabel("Values")
        # ax8.set_title("Reconstr. values")

        # orgDistsArray = X_dist.ravel()
        # reconstrDistsArray = X_reconstructed_dists.ravel()

        # dforig_and_reconstructed_data = pd.DataFrame(list(
        #     zip(orgDistsArray, reconstrDistsArray)), columns=["Orig.", "Reconstr."])
        # create_violin_and_boxplot(
        #     ax=ax9, data=dforig_and_reconstructed_data, saturation=0.5, width=0.2)
        # ax9.set_xlabel("Data sets")
        # ax9.set_ylabel("Distances")
        # ax9.set_title("Reconstr. distances")

        # create_scatterplot(ax=ax10, x=orgDistsArray,
        #                    y=reconstrDistsArray, color="dodgerblue")
        # create_lineplot(ax=ax10, x=[min(np.concatenate([orgDistsArray, reconstrDistsArray])),
        #                             max(np.concatenate([orgDistsArray, reconstrDistsArray]))],
        #                 y=[min(np.concatenate([orgDistsArray, reconstrDistsArray])),
        #                    max(np.concatenate([orgDistsArray, reconstrDistsArray]))], linestyle="dotted")
        # sns.regplot(ax=ax10, x=orgDistsArray, y=reconstrDistsArray,
        #             scatter=False, ci=95, fit_reg=True, color='blue')

        # ax10.set_xlabel("Original distances")
        # ax10.set_ylabel("Reconstr. distances")
        # ax10.set_title("Sheppard plot")
        # ax10.text(0.95 * min(np.concatenate([orgDistsArray, reconstrDistsArray])),
        #           0.95 *
        #           max(np.concatenate([orgDistsArray, reconstrDistsArray])),
        #           "Overal mean standardized RMSE: " + str(round(mean_reconstructionError_dists, 3)), va="center", color="red")

        VioletBlue = sns.color_palette(
            "blend:#03118f,#ecebf4,#572c92", as_cmap=True)
        n_dims_heat = num_pc if num_pc < 20 else n_dims
        create_heatmap(ax=ax11, data=loadings_df_z.T[0:n_dims_heat], cmap=VioletBlue)
        ax11.set_title("Z-normalized loadings")
        ax11.set_xlabel("Variables")
        ax11.set_xticklabels(ax11.get_xticklabels(), rotation=90)

        create_heatmap(
            ax=ax12, data=feature_imp_mat.T[0:n_dims_heat], cmap="Purples")
        ax12.set_title("|Z loadings| * explained variance")
        ax12.set_xlabel("Variables")
        ax12.set_xticklabels(ax12.get_xticklabels(), rotation=90)

        df_relevant_variables_PCA = pd.DataFrame(feature_imp)
        df_relevant_variables_PCA.columns = ["PCAimp"]
        df_relevant_variables_PCA["color"] = "dodgerblue"
        df_relevant_variables_PCA.at[relevant_variables_PCA.index.tolist(
        ), "color"] = "salmon"
        df_relevant_variables_PCA.sort_values(
            by="PCAimp", ascending=False, inplace=True)

        create_barplot(ax=ax13, data=df_relevant_variables_PCA, x=df_relevant_variables_PCA.index,
                       y="PCAimp", palette=df_relevant_variables_PCA["color"].values,
                       order=df_relevant_variables_PCA.index)
        ax13.set_xticklabels(ax13.get_xticklabels(), rotation=90)
        ax13.set_ylabel("Normalized correlation with original data")
        ax13.set_title("Feature importance")

        # df_reconstruction_error_per_variable_values = pd.DataFrame(
        #     reconstruction_error_per_variable_values)
        # df_reconstruction_error_per_variable_values.columns = ["values"]
        # df_reconstruction_error_per_variable_values["color"] = "dodgerblue"
        # df_reconstruction_error_per_variable_values.at[relevant_variables_reconstruction_error_values.index.tolist(
        # ), "color"] = "salmon"
        # df_reconstruction_error_per_variable_values.sort_values(
        #     by="values", ascending=True, inplace=True)
        # create_barplot(ax=ax14, data=df_reconstruction_error_per_variable_values, x=df_reconstruction_error_per_variable_values.index,
        #                y="values", palette=df_reconstruction_error_per_variable_values["color"].values,
        #                order=df_reconstruction_error_per_variable_values.index)
        # ax14.set_xlabel("Variables")
        # ax14.set_ylabel("Standardized RMSE")
        # ax14.set_title("Values reconstruction error")
        # ax14.set_xticklabels(ax14.get_xticklabels(), rotation=90)

        # df_reconstruction_error_per_variable_dist = pd.DataFrame(
        #     reconstruction_error_per_variable_dist)
        # df_reconstruction_error_per_variable_dist.columns = ["dists"]
        # df_reconstruction_error_per_variable_dist["color"] = "dodgerblue"
        # df_reconstruction_error_per_variable_dist.at[relevant_variables_reconstruction_error_dists.index.tolist(
        # ), "color"] = "salmon"
        # df_reconstruction_error_per_variable_dist.sort_values(
        #     by="dists", ascending=True, inplace=True)
        # create_barplot(ax=ax15, data=df_reconstruction_error_per_variable_dist, x=df_reconstruction_error_per_variable_dist.index,
        #                y="dists", palette=df_reconstruction_error_per_variable_dist["color"].values,
        #                order=df_reconstruction_error_per_variable_dist.index)
        # ax15.set_xlabel("Variables")
        # ax15.set_ylabel("Standardized RMSE")
        # ax15.set_title("Distances reconstruction error")
        # ax15.set_xticklabels(ax15.get_xticklabels(), rotation=90)

        # ax00.xaxis.set_visible(False)

    return fig