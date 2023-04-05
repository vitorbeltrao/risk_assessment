'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import argparse
import os
import mlflow

# define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=str, default='all', help='Steps to execute')

_steps = [
    'upload_raw_data',
    'upload_trusted_data',
    'basic_clean',
    'data_check',
    'train_model',
    'test_model',
    'deployment']

def main():
    '''Main file that runs the entire pipeline end-to-end using mlflow
    :param steps: str
    Steps to execute. Default is 'all', which executes all steps
    '''
    # read command line arguments
    args = parser.parse_args()

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = 'risk_assessment'
    os.environ['WANDB_RUN_GROUP'] = 'development'

    # Steps to execute
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    if 'upload_raw_data' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/01_upload_raw_data'
        mlflow.run(project_uri, parameters={'steps': 'upload_raw_data'})

    if 'upload_trusted_data' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/02_upload_trusted_data'
        mlflow.run(project_uri, parameters={'steps': 'upload_trusted_data'})

    if 'basic_clean' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/03_basic_clean'
        mlflow.run(project_uri, parameters={'steps': 'basic_clean'})

    if 'data_check' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/04_data_check'
        mlflow.run(project_uri)

    if 'train_model' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/05_train_model'
        mlflow.run(project_uri, parameters={'steps': 'train_model'})

    if 'test_model' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/06_test_model'
        mlflow.run(project_uri, parameters={'steps': 'test_model'})

    if 'deployment' in active_steps:
        project_uri = 'https://github.com/vitorbeltrao/risk_assessment#components/07_deployment'
        mlflow.run(project_uri, parameters={'steps': 'deployment'})

if __name__ == "__main__":
    # call the main function
    main()
