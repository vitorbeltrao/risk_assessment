'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import os
import json
import hydra
import mlflow
from omegaconf import DictConfig

_steps = [
    'upload_raw_data',
    'upload_trusted_data',
    'basic_clean',
    'data_check',
    'train_model',
    # 'test_model'
]

@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):
    '''Main file that runs the entire pipeline end-to-end using hydra and mlflow

    :param config: (.yaml file)
    file that contains all the default data for the 
    entire machine learning pipeline to run
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    if 'upload_raw_data' in active_steps:
        project_uri = f"{config['main']['components_repository']}/01_upload_raw_data"
        mlflow.run(project_uri)

    if 'upload_trusted_data' in active_steps:
        project_uri = f"{config['main']['components_repository']}/02_upload_trusted_data"
        mlflow.run(project_uri)

    if 'basic_clean' in active_steps:
        project_uri = f"{config['main']['components_repository']}/03_basic_clean"
        mlflow.run(project_uri)

    if 'data_check' in active_steps:
        project_uri = f"{config['main']['components_repository']}/04_data_check"
        mlflow.run(project_uri)

    if 'train_model' in active_steps:
        project_uri = f"{config['main']['components_repository']}/05_train_model"
        mlflow.run(project_uri)


if __name__ == "__main__":
    go()
