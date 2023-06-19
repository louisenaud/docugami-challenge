"""
File:        predict.py
Created by:  Louise Naud
On:          6/15/23
At:          1:06 PM
For project: docugami-challenge
Description:
Usage:
"""
import pickle
import warnings
from typing import Union, Any
import os

import hydra
import mlflow
import numpy as np
from hydra import utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig
import pandas as pd


def read_file(name:Union[str, os.PathLike])->Any:
    with open(utils.to_absolute_path(name), 'rb') as fp:
        f = pickle.load(fp)

    return f

def load_csv_data(name:Union[str, os.PathLike])->Any:
    df = pd.read_csv(name)
    return df

@hydra.main(config_path='../configs/hyperparameters.yaml')
def predict(config:DictConfig):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    df = read_file(config.data)
    X_test = df['clean_title'].values

    model = mlflow.sklearn.load_model(
        utils.to_absolute_path("mlruns/1/b1da1795d808496f8231f7c4fcc3697f/artifacts/kbest"))

    labels_pred = model.predict(X_test)

    print(confusion_matrix(y_test, labels_pred))
    print(metrics.f1_score(y_test, labels_pred, average='weighted'))


if __name__ == '__main__':
    predict()
