from .regression import Regression
from datathon import Data


def dump(model: type, name: str, drop=None) -> None:
    """
    Deploy model prediction to csv
    """
    data_test = Data().test.preprocessing()
    if drop:
        data_test.drop(columns=drop, axis=1)
    pred = model.predict(data_test)
    data_test['pred'] = pred
    data_test.to_csv(f'tries/{name}.csv')
