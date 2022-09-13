import pandas as pd

from typing import Literal
from typing import List
from sklearn.model_selection import train_test_split
from numpy import ndarray


class Data(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        return super(Data, self).__init__(*args, **kwargs)


    @property
    def _constructor(self):
        return Data


    @property
    def train(self):
        return self._upload("train")


    @property
    def test(self):
        return self._upload("test")


    @classmethod
    def _upload(cls, type: Literal["test", "train"]) -> pd.DataFrame:
        data = pd.read_excel(f"bike_{type}.xlsx")
        return cls(data)


    def preprocessing(self) -> pd.DataFrame:
        data = self.copy()
        # Add day column
        data["dy"] = data.dteday.dt.day
        if "casual" in data.columns or "registered" in data.columns:
            droped_columns = ["casual", "registered", "dteday"]
        else:
            droped_columns = ["dteday"]
        return (
            data
            # Set register's column as index
            .set_index("instant")
            # Drop innecesary columns for traning
            .drop(columns=droped_columns)
        )
        
    def train_test_split(self, test_size: float=0.20):
        data = self.copy()
        features = data.columns.drop('cnt')

        mask = int(self.shape[0] * test_size)
        data_train = data[:-mask]
        data_test = data[-mask:]

        X_train = data_train[features]
        y_train = data_train['cnt']
        X_test = data_test[features]
        y_test = data_test['cnt']

        return X_train, X_test, y_train, y_test
