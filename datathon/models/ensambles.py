from typing import List, final
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datathon import Data
from datathon.models import dump


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    RMSE = np.sqrt(mean_squared_error(y_test, predictions, squared=False))
    print('Model Performance')
    print(f'Average Error: {np.mean(errors):0.4f} degrees.')
    print(f'Accuracy = {accuracy:0.2f}%.')
    print(f'RMSE: {RMSE}')
    return accuracy


def train_split(drop: List=[], test_size=0.20):
    data = Data().train.preprocessing()
    mask = int(data.shape[0] * test_size)

    features = data.columns.drop('cnt')

    if drop:
        data.drop(columns=drop)

    data_train = data[:-mask]
    data_test = data[-mask:]

    X_train, y_train = data_train[features], data_train['cnt']
    X_test, y_test = data_test[features], data_test['cnt']

    return X_train, X_test, y_train, y_test
    

def random_forest_regressor():

    # PRIMERA ITERACION
    # max_depth = 9
    # max_features = auto
    # n_estimators = 150
    # param_grid = dict(
    #     n_estimators=[100, 200, 300, 400, 500],
    #     max_features=['auto'],
    #     max_depth=[4, 5, 6, 7, 8, 9, 10],
    # )

    # SEGUNDA ITERACION 
    # max_depth = 40
    # max_features = auto
    # n_estimators = 300
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10, 20, 30, 40, 50],
    #     'max_features': ['auto'],
    #     'n_estimators': [100, 200, 300, 500]
    # }

    # TERCERA ITERACION
    # max_depth = 20
    # max_features = auto
    # n_estimators = 2000
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10, 20, 30, 40, 50, 80, 100],
    #     'max_features': ['auto'],
    #     'n_estimators': [1000, 1500, 2000]
    # }

    rf_reg = RandomForestRegressor()

    param_grid = {
        'bootstrap': [True],
        'max_depth': [5, 6, 7, 8, 9, 10],
        'max_features': ['auto'],
        # 'min_samples_leaf': [3, 4, 5],
        # 'min_samples_split': [8, 10, 12],
        'n_estimators': [50, 100, 200]
    }

    grid_search = GridSearchCV(
        estimator=rf_reg, param_grid=param_grid, cv=4, n_jobs=-1, verbose=2
    )

    # X_train, X_test, y_train, y_test = train_split(
    #     drop=[
    #         # 'yr', 
    #         'mnth', 
    #         # 'holiday', 
    #         # 'weekday', 
    #         # 'workingday',
    #         'atemp', 
    #         # 'dy'
    #     ]
    # )
    data = Data().train.preprocessing()
    X = data.drop(columns=['cnt', 'mnth', 'yr', 'atemp'])
    y = data['cnt']
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=16 
    )
    grid_search.fit(X_train, y_train)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)


    # Best model
    model = RandomForestRegressor(
        max_depth=10, 
        max_features='auto', 
        n_estimators=1000,
        n_jobs=-1,
    )
    
    final_train = Data().train.preprocessing()
    X_final_train, y_final_train = final_train.drop(
        columns=['cnt', 'atemp', 'yr', 'mnth']
    ), final_train['cnt']
    model.fit(X_final_train, y_final_train)
    evaluate(model, X_test, y_test)

    X_final_test = (
        Data().test.preprocessing()
        .drop(columns=['mnth', 'yr', 'atemp'])
    )

    final_pred = model.predict(X_final_test)
    X_final_test['pred'] = final_pred
    X_final_test.to_csv('tries/random_forest_regressor.csv')


    # reparar...
    # dump(
    #     model, 
    #     'random_forest_regressor', 
    #     drop=['atemp', 'yr', 'mnth']
    # )
