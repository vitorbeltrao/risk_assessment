'''
Script to take the raw data from the data lake and do some
transformations to feed the trusted layer of the data lake
and wandb

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import os
import sys
import logging
import wandb
import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
BUCKET_NAME = sys.argv[1]
DESTINATION_RAW_BLOB_PATH = sys.argv[2]
COMPONENT_CURRENT_DIRECTORY = sys.argv[3]
DESTINATION_TRUSTED_BLOB_PATH = sys.argv[4]
# key code for managing the entire infrastructure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'

def download_raw_data(
        bucket_name: str, 
        destination_raw_blob_path: str, 
        component_current_directory: str) -> None:
    '''Function that downloads the files that are in the data lake

    :param bucket_name: (str)
    Name of the respective bucket

    :param destination_raw_blob_path: (str)
    Destination folder from which you want to download the raw data
    '''
    # data download (extract)
    logging.info('Start the download on raw data: SUCCESS')
    data_directory = component_current_directory
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    file_names = list(storage_client.list_blobs(bucket_name, prefix=destination_raw_blob_path))
    file_names = [file.name[len(destination_raw_blob_path):] for file in file_names]

    for file in file_names:
        raw_blob = bucket.blob(destination_raw_blob_path + str(file))
        raw_blob.download_to_filename(data_directory)
    logging.info('Finish the download on raw data: SUCCESS')


def transform_raw_data(component_current_directory: str) -> pd.DataFrame:
    '''Function that does some transformations to the raw data coming from the data lake

    :param component_current_directory: (str)
    Current component path "02" to get some files (.csv) and do some actions

    :return: (dataframe)
    Pandas dataframe
    '''
    # make some transformations (transform)
    logging.info('Start making the transformations on raw data: SUCCESS')
    data_directory = component_current_directory
    pdf = pd.DataFrame(
        columns=[
            'corporation',
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees',
            'exited'])
    for directory in data_directory:
        filenames = os.listdir(directory)
        for each_filename in filenames:
            # some files in this directory are not .csv
            if each_filename.endswith('.csv'):
                current_pdf = pd.read_csv(directory + '/' + each_filename)
                pdf = pdf.append(current_pdf).reset_index(drop=True)


    drop_unnecessary_cols = [col for col in pdf.columns if 'Unnamed' in col]
    pdf.drop(drop_unnecessary_cols, axis=1, inplace=True)
    pdf.drop_duplicates(inplace=True)
    logging.info('Finish making the transformations on raw data: SUCCESS')
    return pdf


def upload_to_storage(
        bucket_name: str,
        data: pd.DataFrame,
        destination_trusted_blob_path: str) -> None:
    '''Function that uploads a dataframe into a google storage bucket

    :param bucket_name: (str)
    Name of the respective bucket

    :param data: (dataframe)
    Dataframe you want to upload

    :param destination_trusted_blob_path: (str)
    Destination folder to upload the already transformed data in the trusted training layer
    '''
    # load transformed datasets (load) to google storage bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    trusted_blob = bucket.blob(destination_trusted_blob_path)
    trusted_blob.upload_from_string(data.to_csv())

    return logging.info(
        'Loading the NEW CSV FILE into the bucket: SUCCESS')


def upload_to_wandb(name_set: str, data: pd.DataFrame) -> None:
    '''Function that uploads data to wandb

    :param name_set: (str)
    Final dataset name you want to name to be saved

    :param data: (dataframe)
    Dataframe you want to upload

    '''
    # load transformed datasets (load) to wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='upload trusted data (transformed data)')
    logging.info('Creating run for risk assessment: SUCCESS')

    artifact = wandb.Artifact(
        name=name_set,
        type='dataset',
        description='Raw dataset transformed with some necessary things to start DS pipeline')

    data.to_csv(name_set, index=False)
    artifact.add_file(name_set)
    run.log_artifact(artifact)
    logging.info(f'Uploaded {name_set} to wandb: SUCCESS\n')


if __name__ == "__main__":
    logging.info('About to start executing of the transform function\n')

    for folder, name in zip(['train_data/', 'test_data/'], ['train_set.csv', 'test_set.csv']):
        # upload train and test data to trusted folder in the bucket
        component_current_directory = [COMPONENT_CURRENT_DIRECTORY]
        download_raw_data(BUCKET_NAME, DESTINATION_RAW_BLOB_PATH + folder, COMPONENT_CURRENT_DIRECTORY)
        trusted_train_set = transform_raw_data(component_current_directory)
        upload_to_storage(BUCKET_NAME, trusted_train_set, DESTINATION_TRUSTED_BLOB_PATH + folder + name)
        upload_to_wandb(name, trusted_train_set)

        for directory in component_current_directory:
            filenames = os.listdir(directory)
            for each_filename in filenames:
                if each_filename.endswith('.csv'):
                    os.remove(each_filename)

    logging.info('Done executing the transform function')
