"""
File:        main.py
Created by:  Louise Naud
On:          6/16/23
At:          3:41 PM
For project: docugami-challenge
Description:
Usage:
"""
import datetime
import json
import logging
import os
import warnings
from pprint import pprint

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import sklearn.base
from hydra import utils
from hydra.utils import instantiate
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors._nearest_centroid import NearestCentroid

from settings import stop_words_all
from src.utils.file_io import load_csv_data


def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []

    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[: -top_n - 1: 1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])

    keywords.sort(key=lambda x: x[1])
    keywords.reverse()
    return_values = []
    for i in keywords:
        return_values.append(i[0])
    return return_values


log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(config):
    pprint(config, indent=4)
    warnings.filterwarnings("ignore")
    # load csv data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # result_exp_folder = os.path.join(dir_path, "results", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    result_exp_folder = os.path.join(dir_path, "results", datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(result_exp_folder, exist_ok=True)
    data_file = os.path.join(dir_path, "data/titles.csv")
    data = load_csv_data(data_file)
    data.drop_duplicates(["title"], inplace=True)
    # get full titles and cleaned titles
    titles = data["clean_title"].values
    full_titles = data["title"].values
    # set-up mlflow tracking
    mlflow.set_tracking_uri(f"file://{utils.get_original_cwd()}/mlruns")
    mlflow.set_experiment(config.mlflow.experiment_name)
    log.info("Instantiating preprocessing pipeline")
    preprocessing_pipeline = hydra.utils.instantiate(config.preprocessing_pipeline, _recursive_=False)
    log.info("Instantiating Clustering Model")
    clusterer = instantiate(config.model)
    # vectorize data
    log.info("Vectorizing titles from papers")
    X = preprocessing_pipeline.fit_transform(titles)
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    with mlflow.start_run():
        clusterer.fit(X)
        labels_pred = clusterer.labels_
        n_labels_pred = len(np.unique(labels_pred))
        mlflow.log_params(clusterer.get_params())
        mlflow.log_artifacts(utils.to_absolute_path("configs"))
        log.info(f"Done clustering for {n_labels_pred} clusters")
        nc = NearestCentroid()
        nc.fit(X, labels_pred)
        centroids = nc.centroids_
        distances = sklearn.metrics.pairwise_distances(X, centroids)
        if isinstance(clusterer, sklearn.cluster.KMeans):
            distances = clusterer.fit_transform(X)
        # elif isinstance(clusterer, sklearn.cluster.SpectralClustering):
        elif isinstance(clusterer, sklearn.cluster.AffinityPropagation):
            cluster_centers_indices = clusterer.cluster_centers_indices_

        log.info("Computing metrics")
        metrics_ = {
            "silhouette": sklearn.metrics.silhouette_score(X, labels_pred),
            "calinski": sklearn.metrics.calinski_harabasz_score(X, labels_pred),
            "davies": sklearn.metrics.davies_bouldin_score(X, labels_pred),
        }
        log.info(metrics_)
        mlflow.log_metrics(metrics_)

        # get topics
        log.info("Getting tags and most representative paper for clusters.")
        data["y"] = labels_pred

        log.info("Create a vectorizer for each cluster")
        vectorizers = []

        for i in range(0, n_labels_pred):
            vectorizers.append(CountVectorizer(min_df=5, max_df=.9, stop_words=list(stop_words_all), lowercase=True))

        log.info("Vectorize data in each cluster")
        vectorized_data = []

        for current_cluster, cvec in enumerate(vectorizers):
            try:
                vectorized_data.append(cvec.fit_transform(data.loc[data['y'] == current_cluster, 'clean_title']))
            except Exception as e:
                log.info("Not enough instances in cluster: " + str(current_cluster))
                vectorized_data.append(None)

        num_topics_per_cluster = 5
        log.info(f"Creating a Latent Dirichlet Allocation for each cluster, with {num_topics_per_cluster} components.")
        lda_models = []
        for i in range(0, n_labels_pred):
            lda = LatentDirichletAllocation(n_components=num_topics_per_cluster, max_iter=10, learning_method='online',
                                            verbose=False, random_state=1)
            lda_models.append(lda)

        log.info("Fit LDA model for each cluster and transform data with it.")
        clusters_lda_data = []
        for current_cluster, lda in enumerate(lda_models):
            if vectorized_data[current_cluster] is not None:
                clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
            # else:
            #     clusters_lda_data.append([])

        log.info("Extract top 5 keywords per topic and select paper that is most representative of the cluster")
        top5keywords = []
        best_papers = []

        for current_vectorizer, lda in enumerate(lda_models):
            if vectorized_data[current_vectorizer] is not None:
                current_keywords = selected_topics(lda, vectorizers[current_vectorizer])
                top5keywords.append(current_keywords[:5])
                # distances to cluster centroid
                if isinstance(clusterer, sklearn.cluster.AffinityPropagation):
                    cluster_center_idx = cluster_centers_indices[current_vectorizer]
                    best_papers.append(str(full_titles[cluster_center_idx]))
                else:
                    indices = np.argwhere(labels_pred == current_vectorizer)
                    current_distances = distances[indices, current_vectorizer]
                    min_idx = np.argmin(current_distances)
                    best_paper_idx = indices[min_idx]
                    best_papers.append(str(full_titles[best_paper_idx]))
            else:
                # in case data is None, we still want to keep track of clusters
                top5keywords.append([])
                best_papers.append([])

        save_dict = {
            "clusterer": str(clusterer.__class__.__name__),
            "n_clusters": n_labels_pred,
            "dim": X.shape[-1],
            "top_words": top5keywords,
            "best_paper": best_papers,
        }
        out_file = os.path.join(
            result_exp_folder,
            f"exp_{config.mlflow.experiment_name}_{save_dict['clusterer']}_n_clusters_{save_dict['n_clusters']}_dim_{save_dict['dim']}.json",
        )
        save_dict |= metrics_
        with open(out_file, "w") as fh:
            json.dump(save_dict, fh)


if __name__ == "__main__":
    main()
