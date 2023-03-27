'''
This .py file is for creating the fixtures

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import sys
import pytest
import pandas as pd
import wandb


@pytest.fixture(scope='session')
def data():
    '''Fixture to generate data to our tests'''
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # download input artifact
    data_path = run.use_artifact('vitorabdo/risk_assessment/clean_data:latest', type='dataset').file()

    if data_path is None:
        pytest.fail('You must provide the csv file')

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data():
    '''Fixture to generate the reference data for non-deterministic tests'''
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # download input artifact
    data_path = run.use_artifact('vitorabdo/risk_assessment/clean_data:v0', type='dataset').file()

    if data_path is None:
        pytest.fail('You must provide the csv file')

    ref_df = pd.read_csv(data_path)
    return ref_df
