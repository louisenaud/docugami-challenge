"""
File:        plotting.py
Created by:  Louise Naud
On:          6/19/23
At:          12:00 PM
For project: docugami-challenge
Description:
Usage:
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly import graph_objects as go

from src.utils.timer import timer
from .cluster_utils import compute_dist_matrix
from .embedder import compute_2d, compute_embeddings, load_embedder

SIZE_POWER = 0.8


def clusters_chart(df: pd.DataFrame, title_col: str = 'cluster_center', hover_col: str = 'cluster_center',
                   size_col: str = 'cluster_size'):
    """
    The clusters_chart function takes a dataframe and plots the clusters in 2D space.
    
    :param df:pd.DataFrame: Specify the dataframe that is passed into the function
    :param title_col:str: Specify the column name of the dataframe that contains titles
    :param hover_col:str: Specify the column that will be used for the hover text
    :param size_col:str: Determine the size of each point in the plot
    :return: A plotly chart of the clusters

    """
    if len(df) == 1:
        st.warning('It is not possible to make plot with only one cluster.')
        return

    with timer('load_embedder inside clusters_chart', disable=False):
        embedder = load_embedder()
    with timer('compute_embeddings inside clusters_chart', disable=False):
        embeddings = compute_embeddings(embedder, df[title_col].tolist())
    with timer('compute_2d inside clusters_chart', disable=False):
        embeddings_2d = compute_2d(embeddings)

    df = df.copy()
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    df = df[df.cluster_size > 1]
    fig = px.scatter(df, x='x', y='y', hover_name=hover_col,
                     size=np.power(df[size_col], SIZE_POWER),
                     hover_data=df.columns,
                     color_discrete_sequence=['#7aaaf7'],
                     width=800, height=600)
    fig.layout.update(showlegend=False)
    st.plotly_chart(fig)


def plot_distance_histogram(max_distance, mean_distance):
    """Plot histograms for max and mean distances between point inside cluster."""

    fig = px.histogram(max_distance[max_distance > 0.01], nbins=50,
                       width=500, height=300,
                       title='Maximum distance')
    st.plotly_chart(fig)

    fig = px.histogram(mean_distance, nbins=50,
                       width=500, height=300,
                       title='Mean distance')
    st.plotly_chart(fig)


def plot_size_histogram(cluster_size: np.array) -> None:
    """
    The plot_size_histogram function takes in a numpy array of cluster sizes and plots a histogram of the data.
        The function uses plotly express to create the histogram, which is then displayed using streamlit's
        st.plotly_chart() method.

    :param cluster_size:np.array: Pass the cluster size array to the function
    :return: A histogram of the cluster sizes
    """
    fig = px.histogram(x=cluster_size, nbins=50,
                       width=500, height=300)
    st.plotly_chart(fig)


def plot_dist_matrix(embeddings: np.array, titles, width: int = 800, height: int = 600):
    """
    The plot_dist_matrix function takes in a list of embeddings and plots the distance matrix between embeddings.

    :param embeddings:np.array: Pass in the embeddings of the images
    :param titles: Label the x and y-axis of the heatmap
    :param width:int: Set the width of the plot
    :param height:int: Set the height of the plot
    :return: A plotly figure object
    """
    dist_matrix = compute_dist_matrix(embeddings)

    fig = go.Figure(data=go.Heatmap(z=dist_matrix, x=titles,
                                    y=titles, colorscale='Viridis_r'))

    fig.update_layout(autosize=False, width=width, height=height,
                      xaxis_showticklabels=False, yaxis_showticklabels=False)

    st.plotly_chart(fig)
