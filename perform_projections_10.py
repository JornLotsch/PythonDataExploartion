#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:44:58 2022

@author: joern
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
from scipy import stats
from sklearn.manifold import MDS, Isomap
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix

from perform_ABCanalysis_5 import ABC_analysis
from plot_PCAresults_grid_2 import plot_results_grid


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


def compute_reconstructionError_iter(data_sc, method):

    nIter = data_sc.shape[1]
    reconstructionError = np.zeros((nIter, 1))
    reconstructionError_dists = np.zeros((nIter, 1))
    data_reconstructed = []
    data_reconstructed_dist = []

    data_sc_dist = distance_matrix(data_sc, data_sc)

    # Compute reconstriction errors
    for i in range(nIter):
        n_dims_i = i + 1
        if method == "ICA":
            model_projection_reconstruction_i = FastICA(
                n_components=n_dims_i)
        else:
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


def determine_n_dims(ndimensions, method, ndimensionsmethod, model, reconstructionError, reconstructionError_dists, indexGE90):
    if ndimensions is not None:
        n_dim = ndimensions
    elif method == "PCA" and ndimensionsmethod == "KaiserGuttman":
        n_dim = np.argmax(model.explained_variance_ < 1)
    elif method == "PCA" and ndimensionsmethod == "ABCeigenvalues":
        ABC_n_eigenvalues = ABC_analysis(data=model.explained_variance_)
        n_dim = len(ABC_n_eigenvalues["Aind"])
    elif method == "PCA" and ndimensionsmethod == "ExplainedVariance":
        n_dim = indexGE90
    elif ndimensionsmethod == "ABCreconstructionErrorValues":
        reconstructionErrorABC = ABC_analysis(
            (sum(reconstructionError) - reconstructionError), PlotIt=False)
        n_dim = len(reconstructionErrorABC["Aind"])
    elif ndimensionsmethod == "ABCreconstructionErrorDistances":
        reconstructionErrorABC_dist = ABC_analysis(
            (sum(reconstructionError_dists) - reconstructionError_dists), PlotIt=False)
        n_dim = len(reconstructionErrorABC_dist["Aind"])
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
    loadings_df_z_abs = loadings_df_z.copy().abs().set_index(loadings_df_z.index)

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


def compute_reconstruction_error_per_variable(data_sc, dataReconstructed):

    reconstruction_error_per_variable_values = np.sqrt(
        ((dataReconstructed - data_sc) ** 2).mean())/data_sc.std()

    dataReconstructed = pd.DataFrame(
        dataReconstructed, columns=data_sc.columns)

    reconstruction_error_per_variable_dist = pd.Series(
        [0.0] * data_sc.shape[1], name="RMSE")
    reconstruction_error_per_variable_dist.index = reconstruction_error_per_variable_values.index

    for i in range(data_sc.shape[1]):
        dist_orig_var_i = pdist(data_sc.iloc[:, i].to_numpy()[
                                :, np.newaxis], metric="euclidean")
        dist_reconstructed_var_i = pdist(dataReconstructed.iloc[:, i].to_numpy()[
                                         :, np.newaxis], metric="euclidean")
        rmse_i = np.sqrt(np.mean((dist_reconstructed_var_i -
                                  dist_orig_var_i) ** 2)) / dist_orig_var_i.std()
        reconstruction_error_per_variable_dist[i] = rmse_i

    return reconstruction_error_per_variable_values, reconstruction_error_per_variable_dist


def identify_relevant_variables_ABC(reconstruction_error_per_variable_values, reconstruction_error_per_variable_dist, feature_imp):

    relevant_variables_reconstruction_error_values = ABC_analysis(sum(reconstruction_error_per_variable_values) -
                                                                  reconstruction_error_per_variable_values)["Aind"]

    relevant_variables_reconstruction_error_dists = ABC_analysis(sum(reconstruction_error_per_variable_dist) -
                                                                 reconstruction_error_per_variable_dist)["Aind"]

    relevant_variables_PCA = ABC_analysis(feature_imp)["Aind"]
    # else:
    #     relevant_variables_PCA = pd.DataFrame(np.zeros((1, 2)),
    #                                           columns = relevant_variables_reconstruction_error_values.columns,  index = ["None"])

    return relevant_variables_reconstruction_error_values, relevant_variables_reconstruction_error_dists, relevant_variables_PCA


def determine_dimensions_to_show(finalDf, n_dims):

    significantPC_y1 = []
    significantPC_y2 = []
    stats_p_y1 = []
    stats_p_y2 = []

    if n_dims > 1:

        for i in range(n_dims):
            if len(np.unique(finalDf.iloc[:, -2])) > 1:
                if len(np.unique(finalDf.iloc[:, -2])) == 2:
                    stats_p_y1.append(stats.mannwhitneyu(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:, -2])]).pvalue)
                else:
                    stats_p_y1.append(stats.kruskal(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:, -2])]).pvalue)
            if len(np.unique(finalDf.iloc[:, -1])) > 1:
                if len(np.unique(finalDf.iloc[:, -2])) == 2:
                    stats_p_y2.append(stats.mannwhitneyu(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:, -1])]).pvalue)
                else:
                    stats_p_y2.append(stats.kruskal(*[group[i].values for name, group in finalDf.groupby(finalDf.iloc[:, -1])]).pvalue)
                    
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


def perform_projections(data, y1 = None, y2 = None, scale=True, method="PCA", showonlyPC12=False, biplot=True, ndimensions=None, minexplainedvarperc=90,
                        ndimensionsmethod="ExplainedVariance"):

    implemented_methods = ["PCA", "ICA", "MDS", "Isomap"]
    if method not in implemented_methods:
        print(method + " not implemented as projection method. Available methods are " +
              str(implemented_methods) + ". Defaulting to PCA.")
        method = "PCA"

    implemented_ndimensionsmethods = ["KaiserGuttman", "ABCeigenvalues",
                                      "ExplainedVariance", "ABCreconstructionErrorValues", "ABCreconstructionErrorDistances"]
    if ndimensionsmethod not in implemented_ndimensionsmethods:
        print(ndimensionsmethod + " not implemented for determining the number of dimensions. Available methods are " +
              str(implemented_ndimensionsmethods) + ". Defaulting to KaiserGuttman.")
        ndimensionsmethod = "KaiserGuttman"

    X_sc, y1, y2, y_comb = projections_clean_data(data=data, y1=y1, y2=y2, scale=scale)

    if ndimensions is not None:
        ndimensions = ndimensions if ndimensions <= data.shape[1] else data.shape[1]
    minexplainedvarperc = minexplainedvarperc if isinstance(
        minexplainedvarperc, int) else 90

    if method == "ICA":
        model_projection = FastICA(n_components=ndimensions)
    elif method == "MDS":
        ndimensions = ndimensions or int(
            2 if ndimensions is None else ndimensions)
        model_projection = MDS(n_components=ndimensions)
    elif method == "Isomap":
        ndimensions = ndimensions or int(
            2 if ndimensions is None else ndimensions)
        model_projection = Isomap(n_components=ndimensions)
    else:
        method = "PCA"
        model_projection = PCA(n_components=ndimensions)

    projections = model_projection.fit_transform(X_sc)
    projectionDf = pd.DataFrame(data=projections)
    finalDf = pd.concat([projectionDf, y1, y2], axis=1)

    if method == "PCA":
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

    reconstructionError, reconstructionError_dists, data_reconstructed, data_sc_dist, \
        data_reconstructed_dist = compute_reconstructionError_iter(
            data_sc=X_sc, method=method)

    n_dims = determine_n_dims(ndimensions=ndimensions, method=method,
                              ndimensionsmethod=ndimensionsmethod, model=model_projection,
                              reconstructionError=reconstructionError, reconstructionError_dists=reconstructionError_dists, indexGE90=indexGE90)
    idxn_dim = n_dims - 1

    if method == "PCA":

        loadings_df, loadings_df_z, feature_imp_mat, df_dotproduct_z_abs, feature_imp_ABCa_n, \
            feature_imp = calculate_PCA_feature_importance(
                data_sc=X_sc, model=model_projection, projections=projections, n_dims=n_dims)

        reconstruction_error_per_variable_values, reconstruction_error_per_variable_dist = \
            compute_reconstruction_error_per_variable(
                data_sc=X_sc, dataReconstructed=data_reconstructed[idxn_dim])

    showPCs, stats_p_y1, stats_p_y2 = determine_dimensions_to_show(
        finalDf=finalDf, n_dims=n_dims)

    if showonlyPC12:
        showPCs = [0, 1]

    relevant_variables_reconstruction_error_values, relevant_variables_reconstruction_error_dists, relevant_variables_PCA = \
        identify_relevant_variables_ABC(reconstruction_error_per_variable_values=reconstruction_error_per_variable_values,
                                        reconstruction_error_per_variable_dist=reconstruction_error_per_variable_dist,
                                        feature_imp=feature_imp)

    results_grid = plot_results_grid(method=method, showPCs=showPCs, finalDf=finalDf, scale=scale, biplot=biplot,
                                     model_projection=model_projection, X=data, X_sc=X_sc, X_dist=data_sc_dist, y1=y1, y2=y2, y_comb=y_comb,
                                     stats_p_y1=stats_p_y1, stats_p_y2=stats_p_y2, reconstructionError=reconstructionError,
                                     reconstructionError_dists=reconstructionError_dists, num_pc=num_pc,
                                     X_reconstructed=data_reconstructed[
                                         idxn_dim], X_reconstructed_dists=data_reconstructed_dist[idxn_dim],
                                     idxn_dim=idxn_dim, num_features=num_features,
                                     indexGE90=indexGE90, pc_list=pc_list, eigenvalues=eigenvalues, n_dims=n_dims,
                                     loadings_df_z=loadings_df_z, feature_imp_mat=feature_imp_mat, feature_imp=feature_imp,
                                     relevant_variables_reconstruction_error_values=relevant_variables_reconstruction_error_values,
                                     relevant_variables_reconstruction_error_dists=relevant_variables_reconstruction_error_dists,
                                     relevant_variables_PCA=relevant_variables_PCA,
                                     reconstruction_error_per_variable_values=reconstruction_error_per_variable_values,
                                     reconstruction_error_per_variable_dist=reconstruction_error_per_variable_dist)

    if method == "PCA":
        return {"Fig": results_grid, "loadings": loadings_df, "projections": finalDf, "reconstruction_errors": reconstructionError}
    else:
        return {"Fig": results_grid, "projections": finalDf}
