"""
File:        data_exploration.py
Created by:  Louise Naud
On:          6/19/23
At:          12:45 PM
For project: docugami-challenge
Description:
Usage:
"""
from collections import defaultdict
from typing import List, Dict, Set

import numpy as np
from loguru import logger


def get_set_of_fields(papers: List[Dict]) -> Set:
    """
    The get_set_of_fields function takes a list of dictionaries as input and returns a set of all the keys in those
    dictionaries.
    This is useful for determining what fields are available to be used in the search function.

    :param papers:List[Dict]: Specify the type of the parameter papers
    :return: A set of the fields in a list of papers
    """
    fields = set()
    for paper in papers:
        for k in paper.keys():
            if k not in fields:
                fields.add(k)
    return fields


def get_counter_of_fields(papers: List[Dict]) -> Dict:
    """
    The get_counter_of_fields function takes in a list of dictionaries and returns a dictionary with the keys being the fields
    and values being how many times that field appears. This is useful for determining which fields are most common.

    :param papers: List[Dict]: Pass in the list of dictionaries
    :return: A dictionary with the number of times each field appears in the papers
    """
    counter = defaultdict(int)

    for paper in papers:
        for k in paper.keys():
            counter[k] += 1
    return counter


def get_papers_with_field(papers: List[Dict], field_name: str):
    """
    The get_papers_with_field function takes a list of papers and a field name as input.
    It returns the number of papers that have the given field, and it also returns
    a list containing only those papers that have the given field.

    :param papers:List[Dict]: Pass in a list of dictionaries
    :param field_name:str: Specify the field name that you want to get papers with
    :return: A tuple of the number of papers with a field, and the list of papers dict with that field
    """
    n_papers = len(papers)
    n_papers_w_field = 0
    papers_w_field = []
    for paper in papers:
        if field_name in paper.keys() and paper[field_name]:
            n_papers_w_field += 1
            papers_w_field.append(paper)
    logger.info(f"{n_papers_w_field} over {n_papers} have the field {field_name}")
    return n_papers_w_field, papers_w_field


def analyze_titles(count_vecs: np.array, word_dict: np.array) -> Dict:
    total_words = np.sum(count_vecs)
    n_words_per_paper = np.sum(count_vecs, axis=-1)
    n_occurence_words = np.sum(count_vecs, axis=0)
    size_dict = n_occurence_words.shape[-1]
    n_occurence_words = n_occurence_words.tolist()
    max_occurence = np.max(n_occurence_words)
    max_word_idx = np.argmax(n_occurence_words)
    # word_dict = count_vectorizer.get_feature_names_out()
    max_word = word_dict[max_word_idx]
    word_occ_indices_sorted = np.argsort(n_occurence_words)
    words_sorted_by_occ = word_dict[word_occ_indices_sorted[::-1]]
    logger.info(f"Word {max_word} appears {max_occurence} times in all documents")
    print(words_sorted_by_occ[0][-10:])
    results = {
        "total_words": total_words,
        "n_words_per_paper": n_words_per_paper.tolist(),
        "n_occurence_words": n_occurence_words,
        "word_occ_indices_sorted": word_occ_indices_sorted,
        "words": word_dict,
    }
    return results


def get_top_k_words(results: Dict, k: int = 20) -> List:
    """
    The get_top_k_words function takes in a dictionary of results and an integer k.
    It returns the top k words from the results dictionary as a list of tuples, where each tuple is (word, occurence).


    :param results: Dict: Pass the results of the count_words function to get_top_k_words
    :param k: int: Specify the number of words to be returned
    :return: A list of tuples containing the top k words and their occurences
    """
    top_k_words = []
    word_dict = results["words"]
    for i in range(k):
        index = results["word_occ_indices_sorted"][0][-i-1]
        current_word = word_dict[index]
        current_occurence = results["n_occurence_words"][0][index]
        logger.info(f"Word #{i} {current_word} appears a total of {current_occurence} times")

        top_k_words.append((current_word, current_occurence))
    return top_k_words


def get_bottom_k_words(results: Dict, k: int = 20) -> List:
    """
    The get_bottom_k_words function takes in a dictionary of results and an integer k.
    It returns the bottom k words that appear the least number of times in all documents.


    :param results: Dict: Pass in the results dictionary that is returned from the get_word_occurences function
    :param k: int: Specify how many words you want to return
    :return: A list of tuples, where each tuple contains a word and its number of occurences
    """
    bottom_k_words = []
    word_dict = results["words"]
    for i in range(k):
        index = results["word_occ_indices_sorted"][0][i]
        current_word = word_dict[index]
        current_occurence = results["n_occurence_words"][0][index]
        # logger.debug(f"Word {current_word} appears a total of {current_occurence} times")

        bottom_k_words.append((current_word, current_occurence))
    return bottom_k_words
