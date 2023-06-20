# Docugami Code / ML Challenge
This is the code for the coding / ML challenge given by Docugami. A description of the experiments is provided in this 
file, in the [Experiments](#experiments) section and the results are presented in a streamlit app.

The first section, [Install and run](#install-and-run), describes how to get the project running, how to run experiments, 
reproduce results and visualize results in the app.

The seconds section, [Repository structure](#repository-structure), describes the organization of the repository, 
and how different libraries are used.

The last section, [Experiments](#experiments), provides a more in-depth description of the experiments, such as choice 
of representation, algorithms...

## Install and run
First, clone the repository:
```shell
git clone https://github.com/louisenaud/docugami-challenge.git
cd docugami-challenge
```
There is a `run.sh` script that has all actions in it, but if you prefer to proceed step by step, you can follow the 
next command lines.
You will need to install dependencies:
```shell
conda create -n docugami-challenge python=3.10
conda activate docugami-challenge
pip install -r requirements.txt
```
Then, we can get data and run pre-processing as well as preliminary experiments:
```shell
mkdir data && cd data
wget https://github.com/midas-network/COVID-19/raw/master/documents/mendeley_library_files/xml_files/mendeley_document_library_2020-03-25.xml
cd ..
# create results directory
mkdir results
# preprocess data
python -m scripts.get_preprocess_data
# run data exploration
python -m scripts.run_data_exploration
# perform preliminary experiments
python -m scripts.run_preliminary_experiments
```
Most experiments can be run and tracked in mlflow with:
```shell
# launch mlflow ui
mlflow ui
python -m main model=kmeans_sk model.n_clusters=3
```

The report is a streamlit app, that you can run as follows:
```shell
streamlit run run_app.py
```
## Repository structure
This repository relies heavily on the [scikit-learn](https://scikit-learn.org/stable/index.html) library for 
vectorization, clustering models and metrics. The experiments are configured with the [Hydra](https://hydra.cc) 
library, and tracked with [MLFlow](https://mlflow.org).
```
├── app                          <- Python files related to the streamlit app 
├── configs                      <- Hydra configuration files
│   ├── model                    <- Model configs
│   ├── preprocessing_pipeline   <- Preprocessing pipeline configs
│   └── config.yaml              <- Main config for training
│
├── data                   <- Project data
│
├── report                 <- Markdown file to be imported in streamlit app
│
├── results                <- Storing results here
│
├── scripts                <- Python and Shell scripts
│
├── src                    <- Source code
│   ├── utils                    <- Utility scripts
│   │
│   ├── data_exploration.py      <- Functions for data exploration
│   ├── topic_modeling.py        <- Functions for topic modeling
│   └── preprocessing.py         <- Functions for text pre-processing
│
├── tests                  <- Tests of any kind
│
├── .env                      <- file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
├── main.py                   <- File for running experiments
├── run.sh                    <- shell script to run everything
├── run_app.py                <- Python script to run streamlit app
└── README.md
```

## Experiments

### Problem Description and choice of metrics
We are given an xml file listing papers on Covid-19 Research until March 25th 2020. 
The goal of this assignment is to:
1. create groups of similar papers, 
2. give either a title or group of tags to each group,
3. find the paper in each group that is the most representative.

There is no ground truth available for this dataset; we are hence going to use clustering, 
and measure the quality of a clustering with metrics that don't require ground truth:
1. Silhouette Coefficient; it can range between -1 and 1, 1 being a score for a 
highly dense clustering. A score around 0 indicates clusters are overlapping.
2. Calinski-Harabasz Index; it is the ratio of the sum of between-clusters 
dispersion and of within-cluster dispersion for all clusters. It measures if the 
clusters are dense and well separated.
3. Davies-Bouldin Index; it computes the average ‘similarity’ between clusters, 
where the similarity is a measure that compares the distance between clusters with
the size of the clusters themselves. It takes values in R+, and values closer to 0 
indicate a better partition.

It is worth noting that all of these scores tend to give better results for convex clusters.

### Plan
The plan is to:
1. Explore data to choose appropriate representations and algorithms
2. Run a set of preliminary experiments with K-Means and ELBOW curves to get a better idea of the number of clusters 
and appropriate dimensionality to get consistent results
3. Run a set of experiments with different clustering models to compare results

### Data Exploration
The data exploration phase yielded different observations:
- the title is the only data that is available for all papers; this is what we are going to use
- the dataset is not big enough to use deep learning, even for fine-tuning; we are hence going to use classical methods 
to represent the paper titles
- the dataset size is also a bit small to consider it statistically relevant, so statistical methods of representation 
(eg TF-IDF) are probably not representative. We are hence going to use a bag-of-words approach
- the most frequent words in the dataset are not useful to our task; these words are directly linked to Covid-19, and 
we know all of these papers are about research in this specific topic. We are hence going to remove them from our vocabulary.

### Preliminary experiments

In these experiments, we are running k-means with dimensionality reduction (to dimensions 2,3,4,5,10,100 and all 
dimensions of the feature vectors), with a different number of clusters, from 0 to 30. 
This allows to draw the ELBOW curve, and determine a good trade-off for the number of clusters.
Looking at the curves, it seems, 
- in dimension 2, a good range of cluster number is 2-6
- in dimension 3, a good range of cluster number is 4-6
- in dimension 4, a good range of cluster number is 5-9
- in dimension 5, a good range of cluster number is 6-10
- in dimension 10, a good range of cluster number is 8-12
- in dimension 100 and above, the ELBOW curve does not present a distinct change in slope; the results are inconclusive
The ELBOW curve gives results that are consistent with the 3 metrics we chose.

### Experiments with different algorithms
All results can be visualized / explored in the streamlit app, at the page `Experiments`.
Our findings are that:
- overall, clusters are not very separable with this representation
- it seems k-means, with dimensionality reduction to dim=2, yields the best results in terms of metrics (silhouette score=0.75)
- while smaller numbers of clusters yield better metrics, the selected tags are more in accordance with the main paper 
with a higher number of clusters.
- Clustering methods that also determine the number of clusters yield a very high number of clusters (~100); there is 
most likely a higher than number 5 or 10  of topics, but not enough data to populate the clusters and hence get a good score for the clustering

### Perspectives
From our experiments, it seems that:
- other fields might be used in order to create more consistent links between papers, such as the authors, the periodical...
  - this involves dealing with missing data on a graph clustering problem
  - the idea would be to modify the constraints of the convex optimization problem for spectral clustering, similarly 
  as in the paper https://papers.nips.cc/paper_files/paper/2014/file/4d6e4749289c4ec58c0063a90deb3964-Paper.pdf

- more data would be beneficial, in order to train more potent models, such as deep neural networks, as they tend to 
de-tangle the feature space and create representations that form more separable clusters.


