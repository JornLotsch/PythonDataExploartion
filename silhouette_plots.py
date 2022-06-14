#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:24:08 2022

@author: joern
"""

# %% imports

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.base import clone
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def silhouette_plots(data, est, max_clusters=6, random_state=None):
    if random_state:
        random_state = 42

    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(15, 20))
        gs0 = gridspec.GridSpec(
            np.ceil((max_clusters-1)/2).astype(int), 2, figure=fig, wspace=.2, hspace=.2)

        for n_clusters in range(2, max_clusters + 1):
            print(n_clusters)

            clusterer = clone(est)
            if hasattr(clusterer, "random_state"):
                clusterer.random_state = random_state
            if hasattr(clusterer, "n_clusters"):
                clusterer.n_clusters = n_clusters
            clusterer.fit(data)
            cluster_labels = clusterer.labels_
            silhouette_avg = silhouette_score(data, cluster_labels)
            print(silhouette_avg)
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            def is_odd(num):
                return num & 0x1
            ax = fig.add_subplot(
                gs0[np.floor((n_clusters - 2)/2).astype(int), is_odd(n_clusters)])
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.viridis(float(i) / n_clusters)
                ax.fill_betweenx(np.arange(
                    y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10  # 10 for the 0 samples
            ax.set_title("Silhouette plot: k = " + str(n_clusters) +
                         ", average Silhouette = " + "{:.3f}".format(silhouette_avg))
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")
            ax.axvline(silhouette_avg, color="salmon", linestyle="dashed")
