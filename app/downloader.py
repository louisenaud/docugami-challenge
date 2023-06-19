"""
File:        downloader.py
Created by:  Louise Naud
On:          6/19/23
At:          12:13 PM
For project: docugami-challenge
Description: Download results.
Usage:
"""
import base64

import pandas as pd
import streamlit as st


def csv_download_link(df: pd.DataFrame, sidebar: bool = False) -> None:
    """
    The csv_download_link function takes a pandas DataFrame and returns an HTML link that, when clicked, downloads the
    DataFrame as a CSV file.
    The function also accepts an optional boolean argument sidebar=False (default) which
    determines whether to display the download link in the main body of the Streamlit app or in its sidebar.

    :param df: pd.DataFrame: Sa pandas dataframe to be converted to csv format
    :param sidebar: bool: Determine whether the download link is displayed in the sidebar or not
    :return: A download link for the dataframe in csv format, into the file 'clusters.csv'
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clusters.csv">Download as csv</a>'

    if sidebar:
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        st.markdown(href, unsafe_allow_html=True)
