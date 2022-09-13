import pandas as pd

from typing import Literal


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
