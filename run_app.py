"""
File:        run_app.py
Created by:  Louise Naud
On:          6/18/23
At:          12:57 PM
For project: docugami-challenge
Description:
Usage:
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.clustering import make_clustering
from app.data import load_data
from app.downloader import csv_download_link
from app.embedder import make_embeddings
from settings import repo_root_path

st.set_page_config(page_title="Covid-19 Paper titles clustering", layout="wide")


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def main():
    session_state = st.session_state
    if "titles" not in st.session_state:
        st.session_state.titles = None
    if "clusters" not in st.session_state:
        st.session_state.clusters = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "df_clusters" not in st.session_state:
        st.session_state.df_clusters = None
    # session_state = st.session_state.get(titles=None, clusters=None,
    #                                  embeddings=None, df_clusters=None)

    st.sidebar.write("""This is an app for clustering of Covid-19 Paper Titles by topics.""")
    st.sidebar.write(
        """
                     Computing text embeddings using Count vectorization,
                     running clustering model, showing results in different ways."""
    )
    st.sidebar.header("Menu")
    mode = st.sidebar.radio(
        "Choose page",
        options=[
            "Data Exploration",
            "Initial Set of Experiments",
            "Experiments",
            "Load data",
            "Clustering",
            "Download clusters",
        ],
    )

    if mode == "Data Exploration":
        data_exploration_page()
    elif mode == "Experiments":
        experiments_page()
    elif mode == "Load data":
        st.title("Explore data more interactively.")
        load_data(session_state)
        with st.expander("Show data"):
            st.write(pd.Series(session_state.titles, name="title"))

        if session_state.titles is not None:
            make_embeddings(session_state)

    elif mode == "Initial Set of Experiments":
        preliminary_experiments_page()
    elif mode == "Clustering":
        st.title("Run experiments interactively")
        if session_state.titles is None:
            st.write("No data. Load data for clustering.")
            return

        make_clustering(session_state)

    elif mode == "Download clusters":
        if session_state.clusters is None:
            st.write("No clusters trained.")
        else:
            st.write("Download titles and corresponding clusters as csv file.")
            df = pd.DataFrame({"title": session_state.titles, "cluster_id": session_state.clusters})
            csv_download_link(df, sidebar=False)


def experiments_page():
    """
    The experiments_page function displays the results of the experiments.

    :return: The results of the experiments
    """
    st.text("Here are the results of the experiments")
    exp_files = []
    labels = []
    files_sorted = sorted(os.listdir(os.path.join(repo_root_path, "results")))
    for file in files_sorted:
        if file.startswith("exp") and file.endswith(".json"):
            labels.append(file.replace(".json", ""))
            exp_files.append(os.path.join(repo_root_path, "results", file))

    option = st.selectbox("Select Experiment to display results", exp_files)
    with open(option) as fh:
        data = json.load(fh)

    st.json(data)

    # st.header("Conclusions")
    # conc_md = read_markdown_file("report/conclusions.md")
    # st.markdown(conc_md)


def preliminary_experiments_page():
    """
    The preliminary_experiments_page function is used to display the results of the initial set of experiments.
    It displays a markdown file with some text explaining what was done, and then it displays an image and a dataframe
    with the metrics obtained for each dimensionality value.

    :return: A page with the results of the initial set of experiments
    """
    st.title("Initial set of experiments")
    md = read_markdown_file("report/initial_set_experiments.md")
    st.markdown(md)

    dim = st.selectbox(
        "Select dimensionality value for vectors after dimensionality reduction",
        (2, 3, 4, 5, 10, 100, "all"),
    )
    if isinstance(dim, int):
        elbow_path = f"results/elbow_curve_dim_{dim}.png"
        metrics_path = f"results/metrics_dim_{dim}.csv"
    else:
        elbow_path = "results/elbow_curve_all_dim.png"
        metrics_path = "results/metrics_all_dim.csv"

    st.text(
        f"Here is the ELBOW curve for vectors of dimension {dim} obtained with K-Means with different number of clusters."
    )
    st.image(elbow_path)
    st.text("And here are the corresponding metrics")
    df_metrics = pd.read_csv(metrics_path)
    st.dataframe(df_metrics)


def data_exploration_page():
    """
    The data_exploration_page function is used to create the data exploration page of the Streamlit app.
    It displays a title, header and Markdown text from files in the report folder. It also displays an image from
    the results' folder.

    :return: A page that contains the data exploration section of the report
    """
    st.title("Data Exploration")
    st.header("Problem Description")

    intro_markdown = read_markdown_file("report/problem_description.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

    st.header("Exploring papers")
    md_papers = read_markdown_file("report/data_exp_papers.md")
    json_file = "results/data_exploration.json"
    if not os.path.exists(json_file):
        st.text(f"File {json_file} does not exist. Running 'python -m scripts.run_data_exploration' to create it.")
        subprocess.run([f"{sys.executable}", "-m", "scripts.run_data_exploration"], capture_output=True)
    st.markdown(md_papers)

    st.header("Exploring words")
    st.markdown(
        "Now we know the title is the only data available for all papers, we are interested in looking at "
        "what words are the most common in all these titles (after removing stopwords):"
    )
    st.image("results/top_words.png")
    st.markdown(
        "We can see that the most common words that appear in almost half of the titles are words that are "
        "directly linked to `coronavirus`. Since we already know all papers are about this specific topic, we are "
        "going to remove them from our dictionary for clustering titles into similar group, as these words are not "
        "likely to help us discriminate between different groups."
    )

    st.header("Choice of representation")
    rep_md = read_markdown_file("report/representation.md")
    st.markdown(rep_md)


if __name__ == "__main__":
    main()
