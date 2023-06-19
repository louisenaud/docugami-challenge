"""
File:        cluster_utils.py
Created by:  Louise Naud
On:          6/19/23
At:          12:02 PM
For project: docugami-challenge
Description:
Usage:
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def compute_cluster_stats(clusters: pd.Series, titles: pd.Series, embeddings: np.array) -> pd.DataFrame:
    """
    The compute_cluster_stats function computes the following statistics for each cluster:
        - cluster_size: number of titles in the cluster
        - max_distance: maximum distance between any two points in the cluster (using inner product)
        - mean_distance: average distance between all pairs of points in the cluster (using inner product)

    :param clusters:pd.Series: Cluster assignments for titles
    :param titles:pd.Series: titles
    :param embeddings:np.array: Embeddings for titles
    :return: A dataframe with the following columns: ['cluster_center','max_distance','mean_distance']
    """
    df_clusters = clusters.value_counts().reset_index()
    df_clusters.columns = ['cluster_id', 'cluster_size']

    centers, max_distances, mean_distances = [], [], []
    for cluster_id in df_clusters.cluster_id:
        cluster_titles = titles[clusters == cluster_id].values
        cluster_embeddings = embeddings[clusters == cluster_id]

        dist_matrix = compute_dist_matrix(cluster_embeddings, metric='inner_product')
        mean_point_distance = dist_matrix.mean(axis=1)
        center_idx = mean_point_distance.argmin()
        upper_triang_dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

        centers.append(cluster_titles[center_idx])
        max_distances.append(dist_matrix.max())
        mean_distances.append(upper_triang_dist.mean())

    df_clusters['cluster_center'] = centers
    df_clusters['max_distance'] = max_distances
    df_clusters['mean_distance'] = mean_distances

    return df_clusters


def compute_dist_matrix(X: np.array, metric: str = 'inner_product') -> np.array:
    """
    The compute_dist_matrix function computes the distance matrix between points of a given dataset.

    :param X:np.array: 2D array with data points
    :param metric:str: Distance metric for scipy.spatial.distance.pdist or 'inner_product'.
                If 'inner_product' then use np.inner instead of pdist which is much faster.
                np.inner could be used instead of cosine distance for normalized vectors
    :return: the pairwise distance matrix
    """
    if X.ndim == 1:
        X = X[None, :]

    if metric == 'inner_product':
        return 1 - np.inner(X, X)
    else:
        return squareform(pdist(X, metric=metric))
