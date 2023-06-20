"""
File:        preprocessing.py
Created by:  Louise Naud
On:          6/15/23
At:          2:21 PM
For project: docugami-challenge
Description:
Usage:
"""
import os
import pickle
import re
import string
from typing import List, Union

import hydra
import pandas as pd
from hydra import utils
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from settings import items_to_clean, repo_root_path
from loguru import logger


# def clean_title(title:str)->str:
#     stop_free = " ".join([i for i in title.lower().split() if i not in stop_words])
#     punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
#     normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
#     return normalized


def get_titles_in_df(
    titles: List[str], out_csv_file: Union[str, os.PathLike] = os.path.join(repo_root_path, "data/titles1.csv")
) -> pd.DataFrame:
    regex_non_alpha = re.compile("[^a-zA-Z\w-]")  # REGEX for chars that are not latin letters or dash

    cleaned_text = []
    for title in titles:
        # get title string into a list of words
        words = title.split()
        # filter each word
        for index, item in enumerate(words):
            # remove words that are not words in latin alphabet
            item = regex_non_alpha.sub("", item)
            # Lowercase the text
            item = item.lower()
            # If the length of item is lower than 3, remove item
            if len(item) < 3:
                item = ""
            # Put item back to the list of words
            words[index] = item
        # remove items in items_to_clean
        cleaned_list = [elem for elem in words if elem not in items_to_clean]
        # reconstruct title as a string for sklearn vectorizer
        cleaned_title = " ".join(cleaned_list)
        # add in list to put in dataframe
        cleaned_text.append(cleaned_title)
    # tokens = tokenize_and_stem(text)
    data = {"title": titles, "clean_title": cleaned_text}
    df = pd.DataFrame(data)
    if out_csv_file:
        logger.info(f"Saving csv file to {out_csv_file}")
        df.to_csv(out_csv_file)
    return df
