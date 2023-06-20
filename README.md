# Docugami Code / ML Challenge

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


## Experiments
