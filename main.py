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

_steps = [
    'upload_raw_data',
    'upload_trusted_data',
    'basic_clean',
    'data_check',
    'train_model',
    'test_model',
    'deployment'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=str, default='all', help='Steps to execute')
    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='MLflow experiment name')
    return parser.parse_args()

def main(args):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = 'risk_assessment'
    os.environ['WANDB_RUN_GROUP'] = 'development'

    # Steps to execute
    steps_par = args.steps
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    # Start MLflow run
    with mlflow.start_run(run_name=args.experiment_name):
        if 'upload_raw_data' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'upload_raw_data')

        if 'upload_trusted_data' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'upload_trusted_data')

        if 'basic_clean' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'basic_clean')

        if 'data_check' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'data_check')

        if 'train_model' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'train_model')

        if 'test_model' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'test_model')

        if 'deployment' in active_steps:
            # Execute the code for this step
            mlflow.log_param('step', 'deployment')

if __name__ == "__main__":
    args = parse_args()
    main(args)
