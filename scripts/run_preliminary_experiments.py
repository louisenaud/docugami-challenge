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
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from settings import repo_root_path
from settings import stop_words_english, stop_words_all
from src.topic_modeling import run_lda
from src.utils.file_io import load_csv_data


def main():
    parser = ArgumentParser(prog='cli')
    csv_default = os.path.join(repo_root_path, "data", "titles1.csv")
    parser.add_argument('--csv-file', default=csv_default, help="Title csv file to load.")

    args = parser.parse_args()

    # load data
    df = load_csv_data(args.csv_file)
    titles = df['title']
    clean_titles = df['clean_title']

    # vectorizer
    vectorizer = CountVectorizer(stop_words=list(stop_words_english))
    vectors = vectorizer.fit_transform(clean_titles)
    total_words = np.sum(vectors)
    n_words_per_paper = np.sum(vectors, axis=-1)
    n_occurence_words = np.sum(vectors, axis=0)
    size_dict = n_occurence_words.shape[-1]
    n_occurence_words = n_occurence_words.tolist()
    max_occurence = np.max(n_occurence_words)
    max_word_idx = np.argmax(n_occurence_words)
    word_dict = vectorizer.get_feature_names_out()
    max_word = word_dict[max_word_idx]
    word_occ_indices_sorted = np.argsort(n_occurence_words)
    print(word_occ_indices_sorted.shape)
    words_sorted_by_occ = word_dict[word_occ_indices_sorted]
    top_20_words = []
    for i in range(20):
        current_word = word_dict[word_occ_indices_sorted[0][-i]]
        print(current_word)
        top_20_words.append(current_word)

    # print(words_sorted_by_occ[0][-20:])
    logger.info(f"Word {max_word} appears {max_occurence} times in all documents")
    # sorted_words = [word_dict[idx] for idx in word_occ_indices_sorted[::-1]]
    # for i in range(20):
    #     print(sorted_words[i])

    # basic clustering
    db_scan = DBSCAN()
    predicted_labels = db_scan.fit_predict(vectors)

    top_words, top_weights = run_lda(vectorizer, clean_titles, 10, 10)
    print(top_words)

    vectorizer2 = CountVectorizer(stop_words=stop_words_all)
    top_words, top_weights = run_lda(vectorizer2, clean_titles, 10, 10)
    print(top_words)


if __name__ == '__main__':
    main()
