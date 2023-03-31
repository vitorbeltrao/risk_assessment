'''
This .py file is for creating the fixtures

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import pytest
import pandas as pd
import wandb
import mlflow
from sklearn.metrics import f1_score


@pytest.fixture(scope='session')
def hist_metrics():
    '''Fixture to generate data to our tests'''
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='check_model_drift')

    # download historical metrics dataset
    previousscores = run.use_artifact('vitorabdo/risk_assessment/historical_metrics:latest', 
                                      type='dataset').file()

    if previousscores is None:
        pytest.fail('You must provide the csv file')

    return previousscores


@pytest.fixture(scope='session')
def newf1score():
    '''Fixture to generate data to our tests'''
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='check_model_drift')

    # download mlflow model
    model_local_path = run.use_artifact('vitorabdo/risk_assessment/final_model_pipe:prod', 
                                        type='pickle').download()
    
    if model_local_path is None:
        pytest.fail('You must provide the pickle file')
    
    # download test dataset
    test_data = run.use_artifact('vitorabdo/risk_assessment/test_set.csv:latest', 
                                 type='dataset').file()
    
    if test_data is None:
        pytest.fail('You must provide the csv file')
    
    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(['exited'], axis=1)
    y_test = test_data['exited']

    # making inference on test set
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)
    f1score = f1_score(y_test, y_pred)
    return f1score