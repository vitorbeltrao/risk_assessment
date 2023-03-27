'''
This .py file is used to clean up the data,
for example removing outliers.

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import sys
import logging
import pandas as pd
import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# config
TRAIN_SET = sys.argv[1]


def clean_data(train_set: str) -> None:
    '''Function to clean up our training dataset to feed the machine
    learning model.

    :param train_set: (str)
    Path to the wandb leading to the training dataset
    '''
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='clean_data')
    artifact = run.use_artifact(train_set, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded trusted data artifact: SUCCESS')

    # clean the train dataset
    df_trusted = pd.read_csv(filepath)
    df_clean = df_trusted.copy()
    logger.info('Train dataset are clean: SUCCESS')

    # upload to W&B
    artifact = wandb.Artifact(
        name='clean_data',
        type='dataset',
        description='Clean dataset after we apply "clean_data" function')

    df_clean.to_csv('df_clean.csv', index=False)
    artifact.add_file('df_clean.csv')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the clean_data function')
    clean_data(TRAIN_SET)
    logging.info('Done executing the clean_data function')