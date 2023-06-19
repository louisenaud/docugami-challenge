"""
File:        embedder.py
Created by:  Louise Naud
On:          6/19/23
At:          12:01 PM
For project: docugami-challenge
Description:
Usage:
"""
"""
Make embeddings.
"""

import numpy as np
import streamlit as st
# from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import tensorflow_text  # pylint: disable=unused-import
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.utils.timer import timer

USE_URL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
USE_URL = 'sentence-transformers/use-cmlm-multilingual'


def make_embeddings(session_state):
    with timer(name='load_embedder', disable=False):
        vectorizer = make_pipeline(
            # HashingVectorizer(stop_words="english"),  # , n_features=50_000
            # TfidfVectorizer(stop_words="english", use_idf=False, min_df=min_df, max_df=max_df),  # , n_features=50_000
            CountVectorizer(stop_words="english", min_df=10, max_df=0.5,lowercase=True),  # , n_features=50_000
            # TfidfTransformer(use_idf=False),
            TruncatedSVD(n_components=20, random_state=0),
            # TruncatedSVD(random_state=0),
            Normalizer(copy=False),
        )
        # _embedder = vectorizer(input=session_state.phrases, max_df=.3, min_df=10, stop_words="english",
        #                             lowercase=True, use_idf=False)

    with timer(name='compute_embeddings', disable=False):
        # session_state.embeddings = compute_embeddings(_embedder, session_state.phrases)
        session_state.embeddings = vectorizer.fit_transform(session_state.phrases)


@st.cache_resource
def load_embedder():
    # return USEEmbedder()
    # return SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
    return TfidfVectorizer()


@st.cache_data
def compute_embeddings(_embedder, phrases):
    embs = _embedder.fit_transform(phrases)
    print(embs.shape)
    return np.array(embs)


@st.cache_data
def compute_2d(embeddings, metric='cosine'):
    model = TSNE(metric=metric)
    return model.fit_transform(embeddings)


# if __name__ == '__main__':
    # model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
    # sentences = ["This is an example sentence", "Each sentence is converted"]
    #
    # embeddings = model.encode(sentences)
    # print(embeddings)
