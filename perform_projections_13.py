#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:44:58 2022

@author: joern
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import stats
from scipy.spatial import distance_matrix

from perform_ABCanalysis_5 import ABC_analysis
from plot_PCAresults_grid_5 import plot_results_grid


def dictfilt(x, y): return dict((i, len(x[i])) for i in x if i in set(y))


def projections_clean_data(data, y1, y2, scale):
    if y1 is None:
        y1 = pd.core.series.Series([0] * data.shape[0])
    else:
        y1 = pd.Series(y1)
    if y2 is None:
        y2 = pd.core.series.Series([0] * data.shape[0])
    else:
        y2 = pd.Series(y2)

    y_comb = []
    for i in range(len(y1)):
        y_comb.append(str(y1[i])+str(y2[i]))

    X_sc = data.copy()

    if scale == True:
        X_sc = pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(X_sc), columns=X_sc.columns)

    return X_sc, y1, y2, y_comb


def perform_pca(data_sc, y1, y2, minexplainedvarperc, ndimensions=None):

    model_projection = PCA(n_components=ndimensions)
    projections = model_projection.fit_transform(data_sc)
    projectionDf = pd.DataFrame(data=projections, columns=data_sc.columns)
    finalDf = pd.concat([projectionDf, y1, y2], axis=1)
   # finalDf.columns =data_sc.columns.tolist() + ["y1"] + ["y2"]

    eigenvalues = model_projection.explained_variance_
    explainedvar = model_projection.explained_variance_ratio_
    num_pc = model_projection.n_components_
    num_features = model_projection.n_features_
    pc_list = ["PC"+str(i) for i in (range(1, num_pc+1))]
    cumvar = np.cumsum(explainedvar)
    if cumvar.max() >= minexplainedvarperc/100:
        indexGE90 = np.argmax(cumvar >= 0.95) + 1
    else:
        indexGE90 = num_pc

    return model_projection, projections, finalDf, eigenvalues, num_pc, num_features, pc_list, indexGE90


def compute_reconstructionError_iter(data_sc):

    nIter = data_sc.shape[1]
    reconstructionError = np.zeros((nIter, 1))
    reconstructionError_dists = np.zeros((nIter, 1))
    data_reconstructed = []
    data_reconstructed_dist = []

    data_sc_dist = distance_matrix(data_sc, data_sc)

    # Compute reconstriction errors
    for i in range(nIter):
        n_dims_i = i + 1
        model_projection_reconstruction_i = PCA(n_components=n_dims_i)
        projections_reconstruction_i = model_projection_reconstruction_i.fit_transform(
            data_sc)
        data_reconstructed_i = model_projection_reconstruction_i.inverse_transform(
            projections_reconstruction_i)

        mean_sq_reconstructionError_i = np.sqrt(
            ((data_reconstructed_i - data_sc) ** 2).mean().mean()) / data_sc.to_numpy().ravel().std()
        reconstructionError[i] = mean_sq_reconstructionError_i

        data_reconstructed_i = data_reconstructed_i.copy()
        data_reconstructed_i_dist = distance_matrix(
            data_reconstructed_i, data_reconstructed_i)

        if (data_reconstructed_i_dist - data_sc_dist != 0).all:
            mean_sq_reconstructionError_i_dist = np.sqrt(((data_reconstructed_i_dist - data_sc_dist)
                                                          ** 2).mean())/data_sc_dist.std()
        else:
            mean_sq_reconstructionError_i_dist = 0
        reconstructionError_dists[i] = mean_sq_reconstructionError_i_dist

        data_reconstructed.append(data_reconstructed_i)
        data_reconstructed_dist.append(
            data_reconstructed_i_dist)

    return reconstructionError, reconstructionError_dists, data_reconstructed, \
        data_sc_dist, data_reconstructed_dist


def determine_n_dims(ndimensions, ndimensionsmethod, model, indexGE90):
    if ndimensions is not None:
        n_dim = ndimensions
    elif ndimensionsmethod == "KaiserGuttman":
        n_dim = np.argmax(model.explained_variance_ < 1)
    elif ndimensionsmethod == "ABCeigenvalues":
        ABC_n_eigenvalues = ABC_analysis(data=model.explained_variance_)
        n_dim = len(ABC_n_eigenvalues["Aind"])
    elif ndimensionsmethod == "ExplainedVariance":
        n_dim = indexGE90
    else:
        n_dim = 2

    if n_dim < 1:
        n_dim = 1

    return n_dim


def calculate_PCA_feature_importance(data_sc, model, projections, n_dims):

    explainedvar = model.explained_variance_ratio_
    loadings = model.components_.T * np.sqrt(model.explained_variance_)
    num_pc = model.n_components_
    pc_list = ["PC"+str(i) for i in (range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings.T)))
    loadings_df_z = (loadings_df.copy()-loadings_df.mean())/loadings_df.std()
    loadings_df['variable'] = data_sc.columns.values
    loadings_df = loadings_df.set_index('variable')
    loadings_df_z['variable'] = data_sc.columns.values
    loadings_df_z = loadings_df_z.set_index('variable')

    diffDims = data_sc.shape[1] - projections.shape[1]
    if data_sc.shape[1] != projections.shape:
        explainedvar = np.append(explainedvar, np.zeros(diffDims))
        projections = np.append(projections, np.zeros(
            [projections.shape[0], diffDims]), axis=1)

    dot_matrix = np.dot(data_sc.T, projections)
    df_dotproduct = pd.DataFrame(dot_matrix, index=loadings_df.index)
    df_dotproduct.columns = pc_list
    df_dotproduct_z = (df_dotproduct.copy() -
                       df_dotproduct.mean())/df_dotproduct.std()
    df_dotproduct_z_abs = df_dotproduct_z.copy().abs().set_index(df_dotproduct_z.index)
    feature_imp_mat = df_dotproduct_z_abs * explainedvar

    feature_imp = feature_imp_mat.iloc[:, :n_dims].sum(axis=1)
    feature_imp.sort_values(axis=0, ascending=False, inplace=True)
    feature_imp_ABC = ABC_analysis(data=feature_imp)
    ABCn = dictfilt(feature_imp_ABC, ("Aind", "Bind"))
    ABCn = [*ABCn.values()]
    feature_imp_ABCa_n = ABCn[0] if ABCn[0] > 0 else sum(ABCn)

    return loadings_df, loadings_df_z, feature_imp_mat, df_dotproduct_z_abs, feature_imp_ABCa_n, feature_imp


def determine_dimensions_to_show(finalDf, n_dims):

    significantPC_y1 = []
    significantPC_y2 = []
    stats_p_y1 = []
    stats_p_y2 = []

    if n_dims >= 1:

        for i in range(n_dims):
            if len(np.unique(finalDf.iloc[:, -2])) > 1:
                if len(np.unique(finalDf.iloc[:, -2])) == 2:
                    stats_p_y1.append(stats.mannwhitneyu(
                        *[group[finalDf.columns[i]].values for name, group in finalDf.groupby(finalDf.iloc[:, -2])]).pvalue)
                else:
                    stats_p_y1.append(stats.kruskal(
                        *[group[finalDf.columns[i]].values for name, group in finalDf.groupby(finalDf.iloc[:, -2])]).pvalue)
            if len(np.unique(finalDf.iloc[:, -1])) > 1:
                if len(np.unique(finalDf.iloc[:, -1])) == 2:
                    stats_p_y2.append(stats.mannwhitneyu(
                        *[group[finalDf.columns[i]].values for name, group in finalDf.groupby(finalDf.iloc[:, -1])]).pvalue)
                else:
                    stats_p_y2.append(stats.kruskal(
                        *[group[finalDf.columns[i]].values for name, group in finalDf.groupby(finalDf.iloc[:, -1])]).pvalue)

        significantPC_y1 = [i for i in range(
            len(stats_p_y1)) if stats_p_y1[i] < 0.05]
        significantPC_y2 = [i for i in range(
            len(stats_p_y2)) if stats_p_y2[i] < 0.05]

    showPCs = np.unique(np.sort(significantPC_y1 + significantPC_y2))
    if len(showPCs) < 1:
        showPCs = [0, 1]
    elif (len(showPCs) < 2):
        if np.min(showPCs) > 0:
            showPCs = [0, showPCs[0]]
        else:
            showPCs = [showPCs[0], 1]
    else:
        showPCs = showPCs[:2]

    return showPCs, stats_p_y1, stats_p_y2


def perform_projections(data, y1=None, y2=None, scale=True, showonlyPC12=False, biplot=True, ndimensions=None, minexplainedvarperc=90,
                        ndimensionsmethod="ExplainedVariance"):

    implemented_ndimensionsmethods = [
        "KaiserGuttman", "ABCeigenvalues", "ExplainedVariance"]
    if ndimensionsmethod not in implemented_ndimensionsmethods:
        print(ndimensionsmethod + " not implemented for determining the number of dimensions. Available methods are " +
              str(implemented_ndimensionsmethods) + ". Defaulting to KaiserGuttman.")
        ndimensionsmethod = "KaiserGuttman"

    X_sc, y1, y2, y_comb = projections_clean_data(
        data=data, y1=y1, y2=y2, scale=scale)

    if ndimensions is not None:
        ndimensions = ndimensions if ndimensions <= data.shape[1] else data.shape[1]
    minexplainedvarperc = minexplainedvarperc if isinstance(
        minexplainedvarperc, int) else 90

    model_projection, projections, finalDf, eigenvalues, num_pc, num_features, pc_list, indexGE90 = perform_pca(
        data_sc=X_sc, y1=y1, y2=y2, minexplainedvarperc=minexplainedvarperc, ndimensions=ndimensions)

    reconstructionError, reconstructionError_dists, data_reconstructed, data_sc_dist, \
        data_reconstructed_dist = compute_reconstructionError_iter(
            data_sc=X_sc)

    n_dims = determine_n_dims(ndimensions=ndimensions, ndimensionsmethod=ndimensionsmethod, model=model_projection, indexGE90=indexGE90)
    idxn_dim = n_dims - 1

    loadings_df, loadings_df_z, feature_imp_mat, df_dotproduct_z_abs, feature_imp_ABCa_n, \
        feature_imp = calculate_PCA_feature_importance(
            data_sc=X_sc, model=model_projection, projections=projections, n_dims=n_dims)

    showPCs, stats_p_y1, stats_p_y2 = determine_dimensions_to_show(
        finalDf=finalDf, n_dims=n_dims)

    if showonlyPC12:
        showPCs = [0, 1]

    relevant_variables_PCA = ABC_analysis(feature_imp)["Aind"]
    least_relevant_variables_PCA = ABC_analysis(feature_imp)["Cind"]

    results_grid = plot_results_grid(ndimensionsmethod=ndimensionsmethod, showPCs=showPCs, finalDf=finalDf, scale=scale, biplot=biplot,
                                     model_projection=model_projection,  X_sc=X_sc, X_dist=data_sc_dist, y1=y1, y2=y2, y_comb=y_comb,
                                     stats_p_y1=stats_p_y1, stats_p_y2=stats_p_y2, reconstructionError=reconstructionError,
                                     reconstructionError_dists=reconstructionError_dists, num_pc=num_pc,
                                     X_reconstructed=data_reconstructed[
                                         idxn_dim], X_reconstructed_dists=data_reconstructed_dist[idxn_dim],
                                     idxn_dim=idxn_dim, num_features=num_features,
                                     indexGE90=indexGE90, pc_list=pc_list, eigenvalues=eigenvalues, n_dims=n_dims,
                                     loadings_df_z=loadings_df_z, feature_imp_mat=feature_imp_mat, feature_imp=feature_imp,
                                     relevant_variables_PCA=relevant_variables_PCA)

    return {"Fig": results_grid, "loadings": loadings_df, "projections": finalDf,
            "reconstructions": data_reconstructed[idxn_dim],
            "relevant_features_pca": relevant_variables_PCA,
            "least_relevant_features_pca": least_relevant_variables_PCA}
