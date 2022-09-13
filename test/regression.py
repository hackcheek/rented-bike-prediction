from datathon import Data
from datathon.models import Regression
from numpy import ndarray


data_test = Data().test.preprocessing()

linear = Regression().linear.predict()
logistic = Regression().logistic.predict()

def dump(predicts: ndarray, name: str) -> None:
    data_test['pred'] = predicts
    data_test.to_csv(f'tries/{name}.csv')


dump(linear, 'linear_regression')
dump(logistic, 'log_regression')
