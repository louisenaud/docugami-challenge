"""
File:        file_io.py
Created by:  Louise Naud
On:          6/15/23
At:          2:10 PM
For project: docugami-challenge
Description:
Usage:
"""
import os
import pickle
from typing import Union, Any, List, Dict

import pandas as pd
import xmltodict
from hydra import utils


def read_file(name: Union[str, os.PathLike]) -> Any:
    with open(utils.to_absolute_path(name), "rb") as fp:
        f = pickle.load(fp)

    return f


def papers_from_xml_file(xml_file_path: Union[str, os.PathLike]) -> List[Dict]:
    """
    The get_docs function takes in a path to an XML file and returns a list of dictionaries, each dictionnary being the
    data of one paper.

    :param xml_file_path:Union[str, os.PathLike]: Specify the type of input that is expected
    :return: A list of dictionaries
    """
    with open(xml_file_path) as fd:
        doc = xmltodict.parse(fd.read(), process_namespaces=True)

    papers = doc["xml"]["records"]["record"]
    return papers


def get_title_text_from_paper(paper_dict: Dict) -> str:
    """
    The get_title_text_from_paper function takes a paper dictionary from xml file as input and
    returns the title text of that paper as a str.

    :param paper_dict:Dict: Specify the type of the parameter
    :return: The title of the paper
    """
    # all papers have the "titles" field
    doc_text = paper_dict["titles"]["title"]

    return doc_text


def get_title_text_all_papers(papers: List[Dict]) -> List[str]:
    """
    The get_text_all_papers function takes a list of dictionaries as input and returns a list of strings.
    The function iterates through the dictionary, appending the abstract, keywords and title to one string for each
    paper.

    :param papers:List[Dict]: Specify the type of the input parameter
    :return: A list of strings, each string is the concatenation of the abstract,
    """
    docs = []
    for dict_doc in papers:
        doc_text = get_title_text_from_paper(dict_doc)
        docs.append(doc_text)
    return docs


def load_csv_data(name: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    The load_csv_data function takes a file name as input and returns the data in that file in a pandas DataFrame.

    :param name: Union[str, os.PathLike]: Specify that the name parameter can be a string or an osPathLike
    :return: A dataframe
    """
    df = pd.read_csv(name)
    return df
