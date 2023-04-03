"""
Model and Data Diagnostics: diagnostic tests related to the model as well as the data.

Author: Vitor Abdo
Date: April/2023
"""

import subprocess

import pandas as pd
import timeit
import os
import json
import pickle
import logging
import wandb
import mlflow

logger = logging.getLogger(__name__)


def model_predictions():
    """
    Get model predictions: read the deployed model and a test dataset, calculate predictions
    :param data: data we use for prediction represented as a panda Dataframe
    :return:
    list containing all predictions
    """
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='test_model')

    # download mlflow model
    model_local_path = run.use_artifact('vitorabdo/risk_assessment/final_model_pipe:prod', type='pickle').download()
    logger.info('Downloaded prod mlflow model: SUCCESS')

    # download test dataset
    test_data = run.use_artifact('vitorabdo/risk_assessment/test_set.csv:latest', type='dataset').file()
    logger.info('Downloaded test dataset artifact: SUCCESS')

    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(['exited'], axis=1)
    y_test = test_data['exited']

    # making inference on test set
    logger.info('Loading model and performing inference on test set')
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)
    return list(y_pred)


def dataframe_summary():
    """
    Get summary statistics
    :return:
    dictionary of statistics (mean, median, std deviation) related to each numerical column
    """
    logger.info('calculate statistics on the data')
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='train_data')
    artifact = run.use_artifact('vitorabdo/risk_assessment/clean_data:latest', type='dataset')
    filepath = artifact.file()
    data = pd.read_csv(filepath)
    X = data.iloc[:, 1:-1]
    means = X.mean()
    medians = X.median()
    std_var = X.std()

    col_stats = {}
    for col in X.columns:
        col_stats[col] = {'mean': means[col], 'median': medians[col], 'std_dev': std_var[col]}

    return col_stats


def missing_data():
    """
    Check for missing data by calculating what percent of each column consists of NA values.
    :return:
    Dictionary with keys corresponding to the columns of the dataset.
    Each element of the dictionary gives the percent of NA values in a particular column of the data.
    """
    logger.info('check for missing data')
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='train_data')
    artifact = run.use_artifact('vitorabdo/risk_assessment/clean_data:latest', type='dataset')
    filepath = artifact.file()
    data = pd.read_csv(filepath)
    missing = data.isna().sum()
    n_data = data.shape[0]
    missing = missing / n_data
    return missing.to_dict()


def execution_time():
    """
    Get timings: calculate timing of training.py and ingestion.py.
    :return:
    list of 2 timing values in seconds
    """
    logger.info('calculate timing for ingestion and training')
    # timing ingestion
    starttime = timeit.default_timer()
    os.system('python3 components/02_upload_trusted_data/upload_trusted_data.py')
    ingestion_timing = timeit.default_timer() - starttime

    # timing training
    starttime = timeit.default_timer()
    os.system('python3 risk_assessment/components/05_train_model/train_model.py')
    training_timing = timeit.default_timer() - starttime

    return [ingestion_timing, training_timing]


def outdated_packages_list():
    """
    Check dependencies: checks the current and latest versions of all the modules that the scripts use
    (the current version is recorded in requirements.txt).
    :return:
    Output a list of dictionaries, one for each package used: the first key will show the name of a Python
    module that is used; the second key will show the currently installed version of that Python module, and
    the third key will show the most recent available version of that Python module:
    [{'module': 'click', 'current': '7.1.2', 'latest': '8.1.3'}, ...]
    """
    logger.info('Check dependencies versions')
    # current version of dependencies
    with open('requirements.txt', 'r') as req_file:
        requirements = req_file.read().split('\n')
    requirements = [r.split('==') for r in requirements if r]
    df = pd.DataFrame(requirements, columns=['module', 'current'])

    # Get outdated dependencies using PIP
    outdated_dep = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf8')
    outdated_dep = outdated_dep.split('\n')[2:]  # the first 2 items are not packages
    outdated_dep = [x.split(' ') for x in outdated_dep if x]
    outdated_dep = [[y for y in x if y] for x in outdated_dep]  # list of [package, current version, latest version]
    outdated_dic = {x[0]: x[2] for x in outdated_dep}  # {package: latest version}
    df['latest'] = df['module'].map(outdated_dic)

    # if we're already using the latest version of a module, we fill latest with this version:
    df['latest'].fillna(df['current'], inplace=True)
    return df.to_dict('records')


if __name__ == '__main__':
    y_pred = model_predictions()
    stats = dataframe_summary()
    missing = missing_data()
    time_check = execution_time()
    outdated = outdated_packages_list()
    pass