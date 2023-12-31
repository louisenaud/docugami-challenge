"""
File:        clusterers.py
Created by:  Louise Naud
On:          6/19/23
At:          12:10 PM
For project: docugami-challenge
Description: Configuring different clustering models.
Usage:
"""

import streamlit as st
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)


def make_clustering_widgets(sidebar=False):
    """Make widgets for selecting clustering algorithm and its parameters."""

    strmlit = st.sidebar if sidebar else st

    clustering_algo = strmlit.selectbox(
        label="Select clustering algorithm",
        options=[
            "Birch",
            "AgglomerativeClustering",
            "KMeans",
            "MiniBatchKMeans",
            "AffinityPropagation",
            "MeanShift",
            "OPTICS",
            "DBSCAN",
        ],
    )

    if clustering_algo == "Birch":
        birch_threshold = strmlit.number_input(
            "Birch clustering threshold", min_value=0.0, max_value=5.0, value=0.75, step=0.01
        )
        clustering_params = {"threshold": birch_threshold}
    elif clustering_algo == "AgglomerativeClustering":
        distance_threshold = strmlit.number_input(
            "Distance_threshold", min_value=0.01, max_value=2.0, value=0.5, step=0.01
        )
        clustering_params = {"distance_threshold": distance_threshold}
    elif clustering_algo == "MeanShift":
        bandwidth = strmlit.slider("bandwidth", min_value=0.0, max_value=2.0, value=0.8, step=0.01)
        if bandwidth == 0:
            bandwidth = None
        clustering_params = {"bandwidth": bandwidth}
    elif clustering_algo == "AffinityPropagation":
        damping = strmlit.slider("damping", min_value=0.5, max_value=1.0, value=0.5, step=0.01)
        clustering_params = {"damping": damping}
    elif clustering_algo == "DBSCAN":
        eps = strmlit.slider("eps", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
        min_samples = strmlit.slider("min_samples", min_value=1, max_value=20, value=5, step=1)
        clustering_params = {"min_samples": min_samples, "eps": eps}
    elif clustering_algo == "OPTICS":
        min_samples = strmlit.slider("min_samples", min_value=1, max_value=50, value=5, step=1)
        clustering_params = {"min_samples": min_samples}
    elif clustering_algo == "MaxDiameterClustering":
        max_distance = strmlit.number_input(
            "Maximum cosine distance between points inside cluster", min_value=0.01, max_value=2.0, value=0.5, step=0.01
        )
        clustering_params = {"max_distance": max_distance}
    elif clustering_algo == "LeaderClustering":
        max_radius = strmlit.number_input(
            "Maximum radius of cluster in terms of cosine distance", min_value=0.01, max_value=1.0, value=0.3, step=0.01
        )
        clustering_params = {"max_radius": max_radius}
    elif clustering_algo == "QTClustering":
        max_radius = strmlit.number_input(
            "Maximum radius of cluster in terms of cosine distance", min_value=0.01, max_value=1.0, value=0.3, step=0.01
        )
        clustering_params = {"max_radius": max_radius}
    else:
        n_clusters = strmlit.slider("Number of clusters", min_value=2, max_value=10, value=5, step=1)
        clustering_params = {"n_clusters": n_clusters}

    return clustering_algo, clustering_params


def get_clustering_model(clustering_algo, params):
    """Get clustering model with given parameters."""

    if clustering_algo == "KMeans":
        model = KMeans(params["n_clusters"], random_state=42)
    elif clustering_algo == "MiniBatchKMeans":
        model = MiniBatchKMeans(params["n_clusters"], random_state=42)
    elif clustering_algo == "Birch":
        model = Birch(n_clusters=None, threshold=params["threshold"], branching_factor=50)
    elif clustering_algo == "AgglomerativeClustering":
        model = AgglomerativeClustering(
            n_clusters=None, affinity="cosine", linkage="complete", distance_threshold=params["distance_threshold"]
        )
    elif clustering_algo == "MeanShift":
        model = MeanShift(bandwidth=params["bandwidth"], cluster_all=False)
    elif clustering_algo == "AffinityPropagation":
        model = AffinityPropagation(damping=params["damping"])
    elif clustering_algo == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    elif clustering_algo == "OPTICS":
        model = OPTICS(min_samples=params["min_samples"])

    return model
