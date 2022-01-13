#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:05:18 2022

@author: joern
"""

def box_and_heatplot(data, title="None"):
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    fig.suptitle("Groups of variables: " + title)
    sns.heatmap(ax=axes[0], data = data)
    ax = sns.violinplot(ax=axes[1], data = data, saturation = 0.5)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    plt.setp(ax.collections, alpha=.3)
    ax = sns.boxplot(ax=axes[1], data = data, fliersize = 0,  width=0.2)
    plt.setp(ax.collections, alpha=.5)
    sns.swarmplot(ax=axes[1], data = data)

    return(fig)
    