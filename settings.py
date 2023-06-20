"""
File:        settings.py
Created by:  Louise Naud
On:          6/18/23
At:          1:22 PM
For project: docugami-challenge
Description:
Usage:
"""


import string
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

XML_URL = "https://github.com/midas-network/COVID-19/raw/master/documents/mendeley_library_files/xml_files/mendeley_document_library_2020-03-25.xml"

repo_root_path = os.path.dirname(os.path.realpath(__file__))

stop_words_english = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)
items_to_clean = {"\n", "\n\n", "\n\n\n", "\n\n\n\n", "", " "}.union(set(stop_words_english))

stop_words_covid = {
    "2020",
    "19",
    "2019",
    "20",
    "covid",
    "coronavirus",
    "wuhan",
    "china",
    "cov",
    "sars",
    "corona",
    "coronaviruses",
    "novel",
    "ncov",
    "sarscov",
    "using",
    "based",
}
stop_words_all = list(stop_words_covid.union(set(ENGLISH_STOP_WORDS)))

punctuation = set(string.punctuation)
