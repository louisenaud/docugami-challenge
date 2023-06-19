"""
File:        get_preprocess_data.py
Created by:  Louise Naud
On:          6/15/23
At:          1:29 PM
For project: docugami-challenge
Description:
Usage:
"""
import os
import urllib.request
from argparse import ArgumentParser
from urllib.parse import urlparse

from loguru import logger

from settings import XML_URL, repo_root_path
from src.preprocessing import get_titles_in_df
from src.utils.file_io import papers_from_xml_file, get_title_text_all_papers


def main():
    # argument parser
    parser = ArgumentParser(prog='Get XML file and pre-process data')
    parser.add_argument('--xml', default=XML_URL, help="URL to the XML file that contains the papers' data", type=str)
    parser.add_argument('--data-dir', default=os.path.join(repo_root_path, "data"), help="Path ")
    parser.add_argument('--out-csv', default=os.path.join(repo_root_path, "data", "titles1.csv"),
                        help="path to csv file to save titles")
    # parse argument
    args = parser.parse_args()
    # get some data
    out_dir = args.data_dir
    a = urlparse(args.xml)
    xml_file_name = os.path.basename(a.path)
    logger.info(f"Retrieving {xml_file_name} at URL {args.xml}")

    out_xml_path = os.path.join(out_dir, xml_file_name)
    logger.info(f"Saving file to {out_xml_path}")
    if not os.path.exists(out_dir):
        logger.info(f"Directory {out_dir} does not exist. Creating it.")
        os.makedirs(out_dir)
    if not os.path.exists(out_xml_path):
        logger.info(f"{xml_file_name} does not exist. Downloading it from given URL at {out_xml_path}.")
        urllib.request.urlretrieve(args.xml, out_xml_path)

    # get papers
    logger.info("Loading downloaded xml file in List of Dict.")
    papers = papers_from_xml_file(out_xml_path)
    logger.info("Getting titles from papers in List of str")
    titles = get_title_text_all_papers(papers)

    # save titles and preprocessed titles in csv
    logger.info(f"Pre-processing titles from papers and saving in csv file at {args.out_csv}")
    _ = get_titles_in_df(titles, out_csv_file=args.out_csv)


if __name__ == '__main__':
    main()
