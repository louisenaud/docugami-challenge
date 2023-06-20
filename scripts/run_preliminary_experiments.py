"""
File:        run_preliminary_experiments.py
Created by:  Louise Naud
On:          6/18/23
At:          2:24 PM
For project: docugami-challenge
Description:
Usage:
"""
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_evaluation import plot

from settings import repo_root_path
from settings import stop_words_english, stop_words_all
from src.topic_modeling import run_lda
from src.utils.file_io import load_csv_data


def score(X, n_clusters):
    model = KMeans(n_init="auto", n_clusters=n_clusters, random_state=1)
    model.fit(X)
    predicted = model.predict(X)
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return {
        "n_clusters": n_clusters,
        "silhouette_score": metrics.silhouette_score(X, predicted),
        "calinski_harabasz_score": metrics.calinski_harabasz_score(X, predicted),
        "davies_bouldin_score": metrics.davies_bouldin_score(X, predicted),
    }


def main():
    parser = ArgumentParser(prog="cli")
    csv_default = os.path.join(repo_root_path, "data", "titles1.csv")
    out_dir = os.path.join(repo_root_path, "results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    parser.add_argument("--csv-file", default=csv_default, help="Title csv file to load.")

    args = parser.parse_args()

    # load data
    df = load_csv_data(args.csv_file)
    titles = df["title"]
    clean_titles = df["clean_title"]

    # vectorizer
    vectorizer = CountVectorizer(stop_words=list(stop_words_all))
    vectors = vectorizer.fit_transform(clean_titles)

    # ELBOW curve with k-means
    kmeans = KMeans(random_state=1, n_init=5)

    # plot elbow curve
    ax = plot.elbow_curve(vectors, kmeans, range_n_clusters=range(1, 30))
    fig = ax.get_figure()
    out_fig_path = os.path.join(out_dir, "elbow_curve_all_dim.png")
    fig.savefig(out_fig_path)
    # get all metrics in dataframe
    df_metrics = pd.DataFrame(score(vectors, n_clusters) for n_clusters in (2, 3, 4, 5, 6, 7, 8, 9, 10))
    df_metrics.set_index("n_clusters", inplace=True)

    (
        df_metrics.style.highlight_max(
            subset=["silhouette_score", "calinski_harabasz_score"], color="lightgreen"
        ).highlight_min(subset=["davies_bouldin_score"], color="lightgreen")
    )
    df_metrics.to_csv(os.path.join(out_dir, "metrics_all_dim.csv"))

    # test different dimensions
    for n_comp in [2, 3, 4, 5, 10, 100]:
        logger.info(f"Computing ELBOW and selected metrics for data in space of dimension {n_comp}")
        kmeans = KMeans(random_state=1, n_init=5)
        svd = TruncatedSVD(n_components=n_comp)
        x_red = svd.fit_transform(vectors)
        # plot elbow curve
        ax = plot.elbow_curve(x_red, kmeans, range_n_clusters=range(1, 30))
        fig = ax.get_figure()
        out_fig_path = os.path.join(out_dir, f"elbow_curve_dim_{n_comp}.png")
        fig.savefig(out_fig_path)
        # get all metrics in dataframe
        df_metrics = pd.DataFrame(
            score(x_red, n_clusters) for n_clusters in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        )
        df_metrics.set_index("n_clusters", inplace=True)

        (
            df_metrics.style.highlight_max(
                subset=["silhouette_score", "calinski_harabasz_score"], color="lightgreen"
            ).highlight_min(subset=["davies_bouldin_score"], color="lightgreen")
        )
        df_metrics.to_csv(os.path.join(out_dir, f"metrics_dim_{n_comp}.csv"))

    # basic clustering
    db_scan = DBSCAN()
    predicted_labels = db_scan.fit_predict(vectors)

    top_words, top_weights = run_lda(vectorizer, clean_titles, 10, 10)
    print(top_words)

    vectorizer2 = CountVectorizer(stop_words=list(stop_words_english))
    top_words, top_weights = run_lda(vectorizer2, clean_titles, 10, 10)
    print(top_words)


if __name__ == "__main__":
    main()
