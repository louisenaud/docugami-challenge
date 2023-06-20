"""
File:        plotting.py
Created by:  Louise Naud
On:          6/19/23
At:          6:58 PM
For project: docugami-challenge
Description:
Usage:
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_clustering(data: np.array, labels: np.array, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=50)
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()
