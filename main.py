"""
File:        main.py
Created by:  Louise Naud
On:          6/16/23
At:          3:41 PM
For project: docugami-challenge
Description:
Usage:
"""
import logging
import os
import pickle
import warnings
from pprint import pprint
import pandas as pd

import clusteval as ce
import hydra
# from src.utils.hydra_sklearn_pipeline import make_pipeline
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import sklearn.base
from hydra import utils
from hydra.utils import instantiate
from sklearn_evaluation import plot
from src.utils.file_io import load_csv_data
from src.utils.printing import print_config
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer

from settings import stop_words_all
log = logging.getLogger(__name__)


def read_file(name):
    with open(utils.to_absolute_path(name), 'rb') as fp:
        f = pickle.load(fp)

    return f


def plot_clusteval(X):
    plt.figure()
    fig, axs = plt.subplots(2, 4, figsize=(25, 10))

    # dbindex
    results = ce.dbindex.fit(X)
    _ = ce.dbindex.plot(results, title='dbindex', ax=axs[0][0], visible=False)
    axs[1][0].scatter(X[:, 0], X[:, 1], c=results['labx']);
    axs[1][0].grid(True)

    # silhouette
    results = ce.silhouette.fit(X)
    _ = ce.silhouette.plot(results, title='silhouette', ax=axs[0][1], visible=False)
    axs[1][1].scatter(X[:, 0], X[:, 1], c=results['labx']);
    axs[1][1].grid(True)

    # derivative
    results = ce.derivative.fit(X)
    _ = ce.derivative.plot(results, title='derivative', ax=axs[0][2], visible=False)
    axs[1][2].scatter(X[:, 0], X[:, 1], c=results['labx']);
    axs[1][2].grid(True)

    # dbscan
    results = ce.dbscan.fit(X)
    _ = ce.dbscan.plot(results, title='dbscan', ax=axs[0][3], visible=False)
    axs[1][3].scatter(X[:, 0], X[:, 1], c=results['labx']);
    axs[1][3].grid(True)

    plt.show()


@hydra.main(config_path='configs',
            config_name='config')
def main(config):
    warnings.filterwarnings("ignore")
    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)
    np.random.seed(40)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_file = os.path.join(dir_path, "data/titles.csv")

    data = load_csv_data(data_file)

    titles = data["clean_title"].values
    # titles = data["title"].values

    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(config.mlflow.experiment_name)
    log.info("Instantiating preprocessing pipeline")
    print(config.preprocessing_pipeline)
    preprocessing_pipeline = hydra.utils.instantiate(
        config.preprocessing_pipeline, _recursive_=False
    )

    log.info("Instantiating Clustering Model")
    clusterer = instantiate(config.model)

    # silhouette_scorer = make_scorer(sklearn.metrics.silhouette_score)
    X = preprocessing_pipeline.fit_transform(titles)
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # elbow curve for experiment
    kmeans = KMeans(random_state=1, n_init=5, )

    # plot elbow curve
    range_n_clusters = list(range(2, 15))
    ax1 = plot.elbow_curve(X, kmeans, range_n_clusters=range_n_clusters)
    #
    # # plot silhouette analysis of provided clusters
    # ax2 = plot.silhouette_analysis(X, kmeans, range_n_clusters=range_n_clusters)
    plt.show()
    #
    # # elbow curve for experiment
    # sc = SpectralClustering(random_state=1, n_init=5)
    #
    # # plot elbow curve
    # # ax1 = plot.elbow_curve(X, sc, range_n_clusters=range_n_clusters)
    #
    # # plot silhouette analysis of provided clusters
    # ax2 = plot.silhouette_analysis(X, sc, range_n_clusters=range_n_clusters)
    # plt.show()

    with mlflow.start_run():

        if isinstance(clusterer, sklearn.base.ClusterMixin):
            log.info(f"Training sklearn model {clusterer.__class__.__name__}")
            clusterer.fit(X)
            labels_pred = clusterer.labels_
            # clusterer.predict(X_test)
            n_labels_pred = len(np.unique(labels_pred))
            mlflow.log_params(clusterer.get_params())
            mlflow.log_artifacts(utils.to_absolute_path("configs"))
            log.info(f"Done clustering for {n_labels_pred} clusters")

            # mlflow.log_metric('silhouette', eval(config.metrics.score)(X.toarray(), labels_pred))

        else:
            results = clusterer.fit(X)
            labels_pred = results['labx']
            # clusterer.predict(X_test)
            n_labels_pred = len(np.unique(labels_pred))
            out_path = os.path.join(dir_path, 'outputs/learned_model_v1')
            clusterer.save(out_path)
            clusterer.plot()
            clusterer.plot_silhouette()
            clusterer.scatter()
            # Plot the dendrogram and make the cut at distance height 60
            # y = clusterer.dendrogram(max_d=20)

            # Cluster labels for this particular cut
            # print(y['labx'])

            # mlflow.sklearn.save_model(
        #     clusterer,
        #     utils.to_absolute_path(
        #         f'models/kbest-{clusterer.__class__.__name__}-{n_labels_pred}-clusters'
        #     ),
        # )
        log.info("Computing metrics")
        metrics_ = {
            "silhouette": sklearn.metrics.silhouette_score(X, labels_pred),
            "calinski": sklearn.metrics.calinski_harabasz_score(X, labels_pred),
            "davies": sklearn.metrics.davies_bouldin_score(X, labels_pred)
        }
        log.info(metrics_)
        mlflow.log_metrics(metrics_)

        # get topics
        vectorizer = CountVectorizer(stop_words=stop_words_all)
        vectors = vectorizer.fit_transform(titles)

        df = pd.DataFrame({
            'title' : data['titles'].values,
            'vector': vectors.tolist(),
                           'cluster': labels_pred.tolist()})

        for cluster in range(n_labels_pred):
            current_cluster_samples = df[df['cluster']==cluster]
            current_cluster_titles = current_cluster_samples['title'].values
            current_cluster_vectors = current_cluster_samples['vector'].values
            # run LDA for topics on vectors

            # get scores

            # get centroid

            # get distances to centroid

            # get element with highest score and shortest distance to centroid



if __name__ == "__main__":
    main()

