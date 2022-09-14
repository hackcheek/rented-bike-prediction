from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from typing import Iterable, List
from numpy import ndarray
from pandas import Series
from datathon import Data


class Regression:
    """
    Usage
    -----
    
    Predict with linear model
    >>> pred = Regression().linear.predict()

    Predict with logistic model
    >>> pred = Regression().logistic.predict()

    pred is a np.array()
    """

    def __init__(self, features: List[str] = ...):
        """
        Preparations

        extract X and y of data for training and X for test

        parameter
        ---------
        features: list of strings
            list of features to eval. It works like a mask
            
        """
        if not features == ...:
            self.data_train = Data().train.preprocessing()[features]
            features.remove('cnt')
            self.data_test = Data().test.preprocessing()[features]
        else:
            self.data_train = Data().train.preprocessing()
            self.data_test = Data().test.preprocessing()

        self.features = self.data_train.columns.drop('cnt')

        self.X_train = self.data_train[self.features]
        self.y_train = self.data_train['cnt']

        self.X_test = self.data_test[self.features]


    @property
    def linear(self, *args, **kwargs):
        self.model = LinearRegression(*args, **kwargs)
        return self


    @property
    def logistic(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)
        return self


    def predict(
        self, X_train=Series, y_train=Series, X_test=Series) -> ndarray:
        """
        this method does the prediction

        parameter
        ---------
        X_train: Iterable
            If provided. this axis will be used in training
        y_tran: Iterable
            If provided. this axis will be used in training
        X_test: Iterable
            If provided. this axis will be used for prediction

        Return
        ------
        numpy array with all predictions by X_test

        X_test is whole the features
        """
        err = "You need chose a model. see Regression().__doc__"
        assert self.model, err

        X_train = self.X_train if X_train.empty else X_train
        X_test = self.X_test if X_test.empty else X_test
        y_train = self.y_train if y_train.empty else y_train

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        return pred
