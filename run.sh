# set-up python env
if conda info --envs | grep -q docugami-challenge; then echo "docugami-challenge already exists"; else conda create -y -n docugami-challenge; fi
conda activate docugami-challenge
pip install -r requirements.txt
# download xml file in created data folder
mkdir data
cd data
wget -c https://github.com/midas-network/COVID-19/raw/master/documents/mendeley_library_files/xml_files/mendeley_document_library_2020-03-25.xml
# shellcheck disable=SC2103
cd ..
# create results directory
mkdir results
# preprocess data
python -m scripts.get_preprocess_data
# run data exploration
python -m scripts.run_data_exploration
# perform preliminary experiments
python -m scripts.run_preliminary_experiments

#mlflow ui

# kmeans experiments
python main.py model=kmeans_sk model.n_clusters=2 mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=3 mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=4 mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=2 preprocessing_pipeline=count mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=3 preprocessing_pipeline=count mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=4 preprocessing_pipeline=count mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=2 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=3 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=kmeans
python main.py model=kmeans_sk model.n_clusters=4 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=kmeans

# spectral clustering experiments
python main.py model=spectral_nn model.n_clusters=2 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=4 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=5 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=8 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=10 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=2 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=3 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=4 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=5 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=8 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=10 preprocessing_pipeline=count mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=2 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=3 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=4 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=5 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=8 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn
python main.py model=spectral_nn model.n_clusters=10 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scnn

python main.py model=spectral_rbf model.n_clusters=2 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=4 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=5 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=8 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=10 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=2 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=3 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=4 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=5 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=8 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=10 preprocessing_pipeline=count mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=2 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=3 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=4 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=5 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=8 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf
python main.py model=spectral_rbf model.n_clusters=10 preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=scrbf

# affinity propagation experiments
python main.py model=affinity_propagation mlflow.experiment_name=affinity_propagation
python main.py model=affinity_propagation preprocessing_pipeline=count mlflow.experiment_name=affinity_propagation
python main.py model=affinity_propagation preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=affinity_propagation

# DBSCAN experiments
python main.py model=dbscan mlflow.experiment_name=dbscan
python main.py model=dbscan preprocessing_pipeline=count mlflow.experiment_name=dbscan
python main.py model=dbscan preprocessing_pipeline=count_tsvd_dim3 mlflow.experiment_name=dbscan


