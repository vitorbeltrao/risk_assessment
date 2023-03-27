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

# config
DATA = sys.argv[1]
REF_DATA = sys.argv[2]


@pytest.fixture(scope='session')
def data(DATA):
    '''Fixture to generate data to our tests'''
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # download input artifact
    data_path = run.use_artifact(DATA).file()

    if data_path is None:
        pytest.fail('You must provide the csv file')

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(REF_DATA):
    '''Fixture to generate the reference data for non-deterministic tests'''
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # download input artifact
    data_path = run.use_artifact(REF_DATA).file()

    if data_path is None:
        pytest.fail('You must provide the csv file')

    ref_df = pd.read_csv(data_path)
    return ref_df
