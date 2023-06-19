# set-up python env
if conda info --envs | grep -q docugami-challenge; then echo "docugami-challenge already exists"; else conda create -y -n docugami-challenge; fi
conda activate docugami-challenge
pip install -r requirements.txt
# download xml file in created data folder
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


python src/train.py model=kmeans_sk model.n_clusters=2
python main.py model=kmeans_sk hydra.mode=MULTIRUN model.n_clusters=2,3,4,5,6,7