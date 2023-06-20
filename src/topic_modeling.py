"""
File:        topic_modeling.py
Created by:  Louise Naud
On:          6/18/23
At:          2:25 PM
For project: docugami-challenge
Description:
Usage:
"""
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def plot_top_words(
    model: sklearn.base.BaseEstimator, feature_names: List[str], n_top_words: int, title: str
) -> Tuple[List, List]:
    """
    The plot_top_words function takes in a fitted model, the feature names (i.e., words),
    the number of top words to plot, and a title for the plot. It returns two lists: one with
    the top n_top_words for each topic and another with their corresponding weights.

    :param model:sklearn.base.BaseEstimator: Pass in the model that is being used
    :param feature_names:List[str]: Provide the names of the features
    :param n_top_words:int: Specify the number of words to be plotted for each topic
    :param title:str: Set the title of the plot
    :return: A list of the top words and their weights for each topic
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    top_words = []
    top_weights = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        top_words.append(top_features)
        top_weights.append(weights)

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    return top_words, top_weights


def run_lda(
    vectorizer: sklearn.feature_extraction.text._VectorizerMixin,
    data_samples: List[str],
    n_components: int,
    n_top_words: int,
) -> Tuple[List, List, np.array]:
    """
    The run_lda function takes a vectorizer, data_samples, n_components and n_top_words as input.
    It then fits the vectorizer to the data samples and creates an LDA model with the specified number of components.
    The function returns a list of top words for each topic in addition to their weights.

    :param vectorizer:sklearn.feature_extraction.text._VectorizerMixin: Specify the vectorizer to be used
    :param data_samples:List[str]: Pass in the list of document titles that we want to perform nmf on
    :param n_components:int: Set the number of topics to be used in the lda model
    :param n_top_words:int: Determine how many words to display for each topic
    :return: A tuple of two lists
    """
    vectors = vectorizer.fit_transform(data_samples)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(vectors)
    print("done in %0.3fs." % (time() - t0))
    feature_names = vectorizer.get_feature_names_out()
    top_words, top_weights = plot_top_words(lda, feature_names, n_top_words, "Topics in LDA model")
    return top_words, top_weights, vectors


def run_nmf(
    vectorizer: sklearn.feature_extraction.text._VectorizerMixin,
    data_samples: List[str],
    n_components: int,
    n_top_words: int,
    init: str = "nndsvda",
) -> Tuple[List, List, np.array]:
    """
    The run_nmf function takes a vectorizer, data samples, number of components (topics),
    number of top words to display for each topic and an initialization method. It then fits the NMF model
    to the data using the given parameters and returns two lists: one containing all top words for each topic
    and another containing their corresponding weights.

    :param vectorizer: sklearn.feature_extraction.text._VectorizerMixin: Specify the vectorizer to be used
    :param data_samples: List[str]: Pass in the list of document titles that we want to perform nmf on
    :param n_components: int: Set the number of topics
    :param n_top_words: int: Determine how many words to display for each topic
    :param init:str: Specify the initialization method for nmf
    :return: A list of top words and a list of weights
    """
    vectors = vectorizer.fit_transform(data_samples)
    t0 = time()
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    )
    t0 = time()
    nmf.fit(vectors)
    print("done in %0.3fs." % (time() - t0))

    feature_names = vectorizer.get_feature_names_out()
    top_words, top_weights = plot_top_words(
        nmf,
        feature_names,
        n_top_words,
        "Topics in NMF model (generalized Kullback-Leibler divergence)",
    )
    return top_words, top_weights, vectors
