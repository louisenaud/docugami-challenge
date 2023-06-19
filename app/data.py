"""
File:        data.py
Created by:  Louise Naud
On:          6/18/23
At:          1:12 PM
For project: docugami-challenge
Description: Load data to be clustered
Usage:
"""

import pandas as pd
import streamlit as st

from src.utils.timer import timer


def load_data(session_state: st.session_state) -> None:
    """
    The load_data function is used to load data from a csv file.
    The user can upload a csv file and choose the size of the sample to use.
    If the sample size is greater than all data, then all data will be used.


    :param session_state: st.session_state: Stored data in the session
    :return: None
    """
    st.header('Load data')
    st.write("""Upload csv file with data.
             Text data for clustering should be in column with name "title" or "clean_title".
             Also choose size of sample to use. If sample size is greater than size of all data,
             then all data will be used.""")

    data_file = st.file_uploader('File with text data', type=['csv'])
    sample_size = st.number_input('Sample size', min_value=500,
                                  max_value=1061, value=1061, step=1)
    load_data_button = st.button('Load data from file')

    if data_file is not None and load_data_button:
        with timer('load data', disable=False):

            df = pd.read_csv(data_file)

            if hasattr(df, 'clean_title'):
                titles = df['clean_title']
            elif hasattr(df, 'title'):
                titles = df['title']
            else:
                st.error('Text data in file should be in column with name "title" or "clean_title".')
                return

            if sample_size < len(titles):
                titles = titles.sample(n=sample_size, replace=False)

            session_state.titles = titles.tolist()
