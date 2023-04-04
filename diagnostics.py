'''
Model and Data Diagnostics: diagnostic tests related to the model as well as the data.

Author: Vitor Abdo
Date: April/2023
'''

import subprocess

import pandas as pd
import timeit
import os
import sys
import pickle
import logging
import wandb

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# # config
# PROD_MODEL_PATH = sys.argv[1]
# TEST_SET = sys.argv[2]
# LABEL_COLUMN = sys.argv[3]
# CLEAN_DATA = sys.argv[4]

# config
PROD_MODEL_PATH = 'prod_deployment_path'
TEST_SET = 'vitorabdo/risk_assessment/test_set.csv:latest'
LABEL_COLUMN = 'exited'
CLEAN_DATA = 'vitorabdo/risk_assessment/clean_data:latest'

run = wandb.init(
    project='risk_assessment',
    entity='vitorabdo',
    job_type='diagnostics')


def model_predictions(prod_model_path: str,
                      test_set: str,
                      label_column: str) -> list:
    '''Get model predictions: read the deployed model and a test dataset, calculate predictions
    
    :param final_model: (pickle)
    Path to "prod_deployment_path" folder that have pkl file with all saved model pipeline

    :param test_set: (str)
    Path to the wandb leading to the test dataset

    :param label_column: (str)
    Column name of the dataset to be trained that will be the label

    :return: (list)
    Returns a list of test data predictions
    '''
    # get mlflow model pkl
    model_path = os.path.join(prod_model_path, 'model.pkl')
    sk_pipe = pickle.load(open(model_path, 'rb'))
    logging.info('Get prod mlflow model: SUCCESS')

    # download test dataset
    test_data = run.use_artifact(test_set, type='dataset').file()
    logging.info('Downloaded test dataset artifact: SUCCESS')

    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop([label_column], axis=1)

    # making inference on test set
    y_pred = sk_pipe.predict(X_test)
    logging.info('Loading model and performing inference on test set: SUCCESS\n')
    return list(y_pred)


def summary_statistics(clean_data: str) -> dict:
    '''Get summary statistics: mean, median and standard deviation

    :param clean_data: (str)
    Clean dataset used in model training

    :return: (dict)
    Dictionary of statistics (mean, median, std deviation) related to each numerical column
    '''
    logging.info('Calculate statistics on the data')

    # download clean_data
    artifact = run.use_artifact(clean_data, type='dataset')
    filepath = artifact.file()
    data = pd.read_csv(filepath)

    # get summary statistics
    X = data.iloc[:, 1:-1]
    means = X.mean()
    medians = X.median()
    std_var = X.std()

    col_stats = {}
    for col in X.columns:
        col_stats[col] = {'mean': means[col], 'median': medians[col], 'std_dev': std_var[col]}

    logging.info('Calculated stats: SUCCESS\n')
    return col_stats


def check_missing_data(clean_data: str) -> dict:
    '''Check for missing data by calculating what percent of each column consists of NA values

    :param clean_data: (str)
    Clean dataset used in model training

    :return: (dict)
    Dictionary with keys corresponding to the columns of the dataset
    Each element of the dictionary gives the percent of NA values in a particular column of the data
    '''
    logging.info('Checking for missing data')

    # download clean_data
    artifact = run.use_artifact(clean_data, type='dataset')
    filepath = artifact.file()
    data = pd.read_csv(filepath)

    # check for missing data
    missing = data.isna().sum()
    n_data = data.shape[0]
    missing = missing / n_data
    logging.info('Check missing data: SUCCESS\n')
    return missing.to_dict()


def execution_time() -> list:
    '''Calculates the execution time of the scripts for monitoring the following files:
    upload_raw_data.py, upload_trusted_data.py, train_model.py

    :return: (list)
    list of 3 timing values in seconds
    '''
    logging.info('Calculate timing for upload data and training model')

    # timing upload_raw_data
    starttime = timeit.default_timer()
    os.system('mlflow run . -P steps=upload_raw_data')
    ingestion_raw_data_timing = timeit.default_timer() - starttime

    # timing upload_trusted_data
    starttime = timeit.default_timer()
    os.system('mlflow run . -P steps=upload_trusted_data')
    ingestion_trusted_data_timing = timeit.default_timer() - starttime

    # timing train_model
    starttime = timeit.default_timer()
    os.system('mlflow run . -P steps=train_model')
    training_timing = timeit.default_timer() - starttime

    logging.info('Calculated timing for scripts: SUCCESS\n')
    return [ingestion_raw_data_timing, ingestion_trusted_data_timing, training_timing]


def outdated_packages_list() -> dict:
    '''Check dependencies: checks the current and latest versions of all the modules that the scripts use
    (the current version is recorded in requirements.txt)

    :return: (dict)
    Output a list of dictionaries, one for each package used: the first key will show the name of a Python
    module that is used; the second key will show the currently installed version of that Python module, and
    the third key will show the most recent available version of that Python module:
    [{'module': 'click', 'current': '7.1.2', 'latest': '8.1.3'}, ...]
    '''
    logging.info('Check dependencies versions')

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
    logging.info('Check dependencies versions: SUCCESS\n')

    return df.to_dict('records')


if __name__ == '__main__':
    logging.info('About to start executing the diagnostics function\n')
    y_pred = model_predictions(PROD_MODEL_PATH, TEST_SET, LABEL_COLUMN)
    stats = summary_statistics(CLEAN_DATA)
    missing = check_missing_data(CLEAN_DATA)
    time_check = execution_time()
    outdated = outdated_packages_list()
    logging.info('Done executing the diagnostics function')