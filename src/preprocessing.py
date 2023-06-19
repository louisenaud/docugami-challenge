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


def get_titles_in_df(titles: List[str], out_csv_file: Union[str, os.PathLike] = os.path.join(repo_root_path,
                                                                                             "data/titles1.csv")) -> pd.DataFrame:
    regex_non_alpha = re.compile('[^a-zA-Z\w-]')  # REGEX for chars that are not latin letters or dash

    cleaned_text = []
    for title in titles:
        # get title string into a list of words
        words = title.split()
        # filter each word
        for index, item in enumerate(words):
            # remove words that are not words in latin alphabet
            item = regex_non_alpha.sub('', item)
            # Lowercase the text
            item = item.lower()
            # If the length of item is lower than 3, remove item
            if len(item) < 3:
                item = ''
            # Put item back to the list of words
            words[index] = item
        # remove items in items_to_clean
        cleaned_list = [elem for elem in words if elem not in items_to_clean]
        # reconstruct title as a string for sklearn vectorizer
        cleaned_title = " ".join(cleaned_list)
        # add in list to put in dataframe
        cleaned_text.append(cleaned_title)
    # tokens = tokenize_and_stem(text)
    data = {
        "title": title,
        "clean_title": cleaned_text
    }
    df = pd.DataFrame(data)
    if out_csv_file:
        logger.info(f"Saving csv file to {out_csv_file}")
        df.to_csv(out_csv_file)
    return df


def read_file(file):
    return open(utils.to_absolute_path(file)).read().split("\n")


def save_file(file, name):
    with open(utils.to_absolute_path(name), "wb") as fp:
        pickle.dump(file, fp)


class RegexReplacer(object):
    def __init__(self):
        self.patterns = [
            (r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '<url>'),
            (r'@\w+', '<user>'),
            (r'&\w+', '')  # Replace "&..." with ''
        ]
        self.patterns = [(re.compile(regrex), repl) for (regrex, repl) in
                         self.patterns]

    # Replace the words that match the patterns with replacement words
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


class ProcessText:

    def __init__(self):
        self.tknz = TweetTokenizer()
        self.replacer = RegexReplacer()
        self.stopwords = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.lemmatizer = WordNetLemmatizer()

    def normalize(self, doc):
        for i in range(len(doc)):
            # Tokenize with replacement
            doc[i] = self.tknz.tokenize(self.replacer.replace(doc[i]))

            # Filter stopwords, punctuations, and lowercase
            doc[i] = [w.lower() for w in doc[i] if w not in self.punc and w not in self.stopwords]

            # Stem words

            doc[i] = [self.lemmatizer.lemmatize(w, pos='v') for w in doc[i]]

            # concat
            doc[i] = ' '.join(w for w in doc[i])

        return doc


@hydra.main(config_path='../configs/data_preprocess.yaml')
def main(config):
    train_text = read_file(config.data.text.train)
    val_text = read_file(config.data.text.val)
    test_text = read_file(config.data.text.test)
    train_label = read_file(config.data.label.train)
    val_label = read_file(config.data.label.val)

    process_text = ProcessText()

    train_text = process_text.normalize(train_text)
    val_text = process_text.normalize(val_text)
    test_text = process_text.normalize(test_text)
    train_label = process_text.normalize(train_label)
    val_label = process_text.normalize(val_label)

    save_file(train_text, config.processed_data.text.train)
    save_file(val_text, config.processed_data.text.val)
    save_file(test_text, config.processed_data.text.test)
    save_file(train_label, config.processed_data.label.train)
    save_file(val_label, config.processed_data.label.val)


if __name__ == '__main__':
    main()
