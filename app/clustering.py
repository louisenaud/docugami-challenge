"""
File:        clustering.py
Created by:  Louise Naud
On:          6/19/23
At:          12:10 PM
For project: docugami-challenge
Description: Make clustering and show results.
Usage:
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.timer import timer
from .cluster_utils import compute_cluster_stats, compute_dist_matrix
from .clusterers import get_clustering_model, make_clustering_widgets
from .plotting import clusters_chart, plot_dist_matrix, plot_distance_histogram, plot_size_histogram


def make_clustering(session_state: st.session_state):
    """
    The make_clustering function is the main function for the clustering section of this app.
    It allows users to select a clustering algorithm and its parameters, then displays results
    of that algorithm's performance on the dataset. The user can also view individual clusters
    and their contents.

    :param session_state:st.session_state: Make the state of the app persistent
    :return: The clustering algorithm and its parameters
    """
    st.header("Clustering parameters")
    st.write(
        """Choose clustering algorithm and its parameters.
             Click 'Make clusters' button to make clusters with this parameters."""
    )
    clustering_algo, clustering_params = make_clustering_widgets()

    if cluster_button := st.button("Make clusters"):
        with timer(name="clustering", disable=False):
            start_time = time.time()
            model = get_clustering_model(clustering_algo, clustering_params)
            session_state.clusters = run_model(model, session_state.embeddings)
            st.write(f"Clustering time: {time.time() - start_time:.2f}s.")

        with timer(name="clusters stats", disable=False):
            session_state.df_clusters = compute_cluster_stats(
                session_state.clusters, pd.Series(session_state.titles), session_state.embeddings
            )

    if session_state.clusters is not None:
        st.header("Clustering results")
        st.write(f"Total number of titles: {len(session_state.titles)}")
        st.write(f"Number of clusters: {len(session_state.df_clusters)}")

        with st.expander("All clusters"):
            with timer(name="all clusters", disable=False):
                st.write(
                    """Table with all clusters. It contains cluster id, cluster size,
                         cluster center (phrase with minumum average cosine distance to
                         all other titles in cluster), maximum and mean cosine distance
                         between all titles in cluster."""
                )
                st.write(session_state.df_clusters)

        with st.expander("Chart with clusters"):
            with timer(name="clusters chart", disable=False):
                st.write(
                    """Plot all cluster centers using dimensionality reduction (t-SNE).
                         One point corresponds to one cluster. Size of point reflects
                         size of cluster."""
                )
                clusters_chart(session_state.df_clusters)

        with st.expander("Distance between points in cluster"):
            with timer(name="distance histograms", disable=False):
                st.write(
                    """Histograms with maximum and mean cosine distance between points
                         inside clusters."""
                )
                plot_distance_histogram(session_state.df_clusters.max_distance, session_state.df_clusters.mean_distance)

        with st.expander("Clusters size histogram"):
            with timer(name="clusters size histogram", disable=False):
                plot_size_histogram(session_state.df_clusters.cluster_size)

        show_cluster(session_state)


@st.cache_data
def run_model(_model, data):
    """Run clustering model."""
    return pd.Series(_model.fit_predict(data))


def show_cluster(session_state):
    """Show info and titles from single cluster."""

    st.header("Show cluster")
    st.write("""Show titles and info for single cluster.""")

    sort_type = st.radio("Sort clusters", options=["by size", "by id"])
    if sort_type == "by size":
        df_clusters = session_state.df_clusters
    elif sort_type == "by id":
        df_clusters = session_state.df_clusters.sort_values("cluster_id").reset_index(drop=True)

    cluster_idx = st.number_input(
        label="Show cluster #", min_value=0, max_value=len(session_state.df_clusters) - 1, value=0, step=1
    )
    cluster_info = df_clusters.loc[cluster_idx]
    cluster_id = cluster_info.cluster_id

    st.write(f"Cluster id: {cluster_id}")
    st.write(f"Cluster size: {cluster_info.cluster_size}")
    st.write(f"Cluster center: {cluster_info.cluster_center}")
    st.write(f"Maximum distance between points inside cluster: {cluster_info.max_distance:.2f}")
    st.write(f"Mean distance between points inside cluster: {cluster_info.mean_distance:.2f}")

    titles = pd.Series(session_state.titles, name="phrase")
    cluster_titles = titles[session_state.clusters == cluster_id]
    cluster_embeddings = session_state.embeddings[session_state.clusters == cluster_id]
    st.write("Phrases from cluster:")
    st.table(cluster_titles.sort_values().reset_index(drop=True))

    with st.expander("Distance matrix between points"):
        st.write("""Plot pairwise cosine distance between all titles in cluster.""")
        plot_dist_matrix(cluster_embeddings, cluster_titles, width=800, height=600)

    with st.expander("Distance between points statistics"):
        st.write(
            """Statistics for all pairwise cosine distance between titles in cluster:
                 min, max, mean, std and percentiles."""
        )
        dist_matrix = compute_dist_matrix(cluster_embeddings, metric="inner_product")
        upper_triang_dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        st.write(pd.Series(upper_triang_dist).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict())
