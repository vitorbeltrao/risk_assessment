'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import argparse
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
    'test_model']

def main():
    '''Main file that runs the entire pipeline end-to-end using mlflow
    '''

    # Steps to execute
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    if 'upload_raw_data' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/01_upload_raw_data/main.py', 
                   parameters={'steps': 'upload_raw_data'})

    if 'upload_trusted_data' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/02_upload_trusted_data/main.py', 
                   parameters={'steps': 'upload_trusted_data'})

    if 'basic_clean' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/03_basic_clean/main.py', 
                   parameters={'steps': 'basic_clean'})

    if 'data_check' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/04_data_check/main.py', 
                   parameters={'steps': 'data_check'})

    if 'train_model' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/05_train_model/main.py', 
                   parameters={'steps': 'train_model'})

    if 'test_model' in active_steps:
        project_uri = "https://github.com/vitorbeltrao/risk_assessment.git"
        mlflow.run(project_uri, entry_point='components/06_test_model/main.py', 
                   parameters={'steps': 'test_model'})


if __name__ == "__main__":
    # parse command line arguments
    args = parser.parse_args()

    # pass command line arguments to the main function
    main()
