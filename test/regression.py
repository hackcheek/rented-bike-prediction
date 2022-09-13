"""
Regression models testing
"""
import plotly.express as px
import numpy as np

from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import Iterable

from datathon import Data
from datathon.models import Regression


def main() -> None:
    
    Xy_split = Data().train.preprocessing().train_test_split()
    pred_kwargs = dict(
        X_train=Xy_split[0],
        X_test=Xy_split[1],
        y_train=Xy_split[2],
    )
    y_test = Xy_split[3]
    
    # Prediction 
    linear_pred = Regression().linear.predict(**pred_kwargs)
    # log_pred = Regression().logistic.predict(**pred_kwargs)
    
    print(Data().test.preprocessing().shape)

    # linear_confusion_matrix = conf_matrix(y_test, linear_pred)
    # log_confusion_matrix = conf_matrix(y_test, log_pred)

    # Evaluation model


def dump(predicts: ndarray, name: str) -> None:
    data_test = Data().test.preprocessing()
    data_test['pred'] = predicts
    data_test.to_csv(f'tries/{name}.csv')


def default_linear():
    linear = Regression().linear.predict()
    dump(linear, 'linear_regression')


def default_log():
    logistic = Regression().linear.predict()
    dump(logistic, 'log_regression')


def conf_matrix(y_test: Iterable, y_pred: Iterable, plot=True):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cnf_matrix)
    fig.show()
    return cnf_matrix


if __name__ == '__main__':
    main()
