"""
File:        run_data_exploration.py
Created by:  Louise Naud
On:          6/19/23
At:          1:57 PM
For project: docugami-challenge
Description:
Usage:
"""
import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from settings import repo_root_path, stop_words_english
from src.data_exploration import get_counter_of_fields, get_papers_with_field, analyze_titles, get_top_k_words, \
    get_bottom_k_words
from src.utils.file_io import load_csv_data
from src.utils.file_io import papers_from_xml_file


def main():
    parser = ArgumentParser(prog='cli', description="Script to perform data exploration and save results in json.")
    xml_default = os.path.join(repo_root_path, "data", "mendeley_document_library_2020-03-25.xml")
    out_dir = os.path.join(repo_root_path, "results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_default = os.path.join(out_dir, "data_exploration.json")
    parser.add_argument('--xml-file', default=xml_default, help="Path to xml files with papers.")
    parser.add_argument('--out-json', default=out_default, help="Path to json result file.")
    csv_default = os.path.join(repo_root_path, "data", "titles1.csv")
    parser.add_argument('--csv-file', default=csv_default, help="Title csv file to load.")

    args = parser.parse_args()

    # get papers list
    papers = papers_from_xml_file(args.xml_file)
    counter = get_counter_of_fields(papers)
    keys_all_papers = [k for k, v in counter.items() if v == len(papers)]
    paper_analytics = {
        "n_papers": len(papers)
    }

    for k in keys_all_papers:
        n_papers, _ = get_papers_with_field(papers, k)
        paper_analytics[k] = n_papers
        # logger.info(f"Number of papers with ")

    df = load_csv_data(args.csv_file)
    titles = df['title']
    clean_titles = df['clean_title']

    # vectorizer
    vectorizer = CountVectorizer(stop_words=list(stop_words_english))
    vectors = vectorizer.fit_transform(clean_titles)
    words = vectorizer.get_feature_names_out()
    res_words = analyze_titles(vectors, words)
    top_k_words = get_top_k_words(res_words, k=25)
    bot_k_words = get_bottom_k_words(res_words, k=25)
    paper_analytics["top_words"] = top_k_words
    paper_analytics["bot_words"] = bot_k_words
    # bar plot top-k words
    words = [w[0] for w in top_k_words]
    occurences = [w[1] for w in top_k_words]
    ind = np.arange(len(words))
    fig = plt.figure(figsize=(15, 7))

    # creating the bar plot
    plt.bar(words, occurences)

    plt.xlabel("Courses offered")
    plt.ylabel('Occurrences')
    plt.title("Top 25 words per occurrence")
    plt.xticks(ind, words, rotation=30, ha='right')
    # plt.show()

    # plt.show()
    fig_out_path = os.path.join(out_dir, "top_words.png")
    plt.savefig(fig_out_path, dpi=500)

    with open(args.out_json, "w") as fh:
        json.dump(paper_analytics, fh)

    # from pprint import pprint
    # pprint(counter, indent=4)

    # # load data
    # df = load_csv_data(args.xml_file)
    # titles = df['title']
    # clean_titles = df['clean_title']
    #
    # # vectorizer
    # vectorizer = CountVectorizer(stop_words=list(stop_words_english))
    # vectors = vectorizer.fit_transform(clean_titles)
    # total_words = np.sum(vectors)
    # n_words_per_paper = np.sum(vectors, axis=-1)
    # n_occurence_words = np.sum(vectors, axis=0)
    # size_dict = n_occurence_words.shape[-1]
    # n_occurence_words = n_occurence_words.tolist()
    # max_occurence = np.max(n_occurence_words)
    # max_word_idx = np.argmax(n_occurence_words)
    # word_dict = vectorizer.get_feature_names_out()
    # max_word = word_dict[max_word_idx]
    # word_occ_indices_sorted = np.argsort(n_occurence_words)
    # print(word_occ_indices_sorted.shape)
    # words_sorted_by_occ = word_dict[word_occ_indices_sorted]
    # top_20_words = []
    # for i in range(20):
    #     current_word = word_dict[word_occ_indices_sorted[0][-i]]
    #     print(current_word)
    #     top_20_words.append(current_word)
    #
    # # print(words_sorted_by_occ[0][-20:])
    # logger.info(f"Word {max_word} appears {max_occurence} times in all documents")
    # # sorted_words = [word_dict[idx] for idx in word_occ_indices_sorted[::-1]]
    # # for i in range(20):
    # #     print(sorted_words[i])
    #
    # # basic clustering
    # db_scan = DBSCAN()
    # predicted_labels = db_scan.fit_predict(vectors)
    #
    # top_words, top_weights = run_lda(vectorizer, clean_titles, 10, 10)
    # print(top_words)
    #
    # vectorizer2 = CountVectorizer(stop_words=stop_words_all)
    # top_words, top_weights = run_lda(vectorizer2, clean_titles, 10, 10)
    # print(top_words)
    #


if __name__ == '__main__':
    main()
