'''
This .py file is for creating the fixtures

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import pytest
import pandas as pd
import wandb


@pytest.fixture(scope='session')
def data():
    '''fixture to generate data to our tests'''
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact('vitorabdo/risk_assessment/clean_data:latest').file()

    if data_path is None:
        pytest.fail('You must provide the csv file')

    df = pd.read_csv(data_path)
    return df