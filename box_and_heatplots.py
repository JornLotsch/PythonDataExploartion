#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:05:18 2022

@author: joern
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd


def box_and_heatplot(data, title=None, scale=False, **kwargs):

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    Title = ("Groups of variables: " +
             title) if title != None else "Variables"
    fig.suptitle(Title)
    if len(data.columns) > 0:
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            data_scaled = min_max_scaler.fit_transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
            ax = sns.heatmap(ax=axes[0], data=data_scaled, **kwargs)
        else:
            ax = sns.heatmap(ax=axes[0], data=data, **kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax = sns.violinplot(ax=axes[1], data=data, saturation=0.5, linewidth= 0.1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.collections, alpha=.5)
        sns.swarmplot(ax=axes[1], data=data)
    else:
        ax = fig.add_subplot(1, 2, 1)
        ax.text(0.5, 0.5, "No data", fontsize=24)

    return(fig)
