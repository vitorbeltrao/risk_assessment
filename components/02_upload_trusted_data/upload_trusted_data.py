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


def transform_raw_data(
        bucket_name: str,
        destination_raw_blob_path: str,
        component_current_directory: str,
        destination_trusted_blob_path: str,
        name_set: str) -> None:
    '''Function that takes all datasets that are in the raw layer
    of the data lake and applies two transformations: union of
    datasets and drop duplicate data. After that, the transformed
    data set is uploaded to the trusted layer of google storage
    and to the wandb

    :param bucket_name: (str)
    Name of the respective bucket

    :param destination_raw_blob_path: (str)
    Destination folder from which you want to download the raw training data

    :param component_current_directory: (str)
    Current component path "02" to get some files (.csv) and do some actions

    :param destination_trusted_blob_path: (str)
    Destination folder to upload the already transformed data in the trusted training layer

    :param name_set: (str)
    Final dataset name you want to name to be saved
    '''
    # 1. data download (extract)
    logging.info('Start the download on raw data: SUCCESS')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_names = list(
        storage_client.list_blobs(
            bucket_name,
            prefix=destination_raw_blob_path))
    file_names = [file.name[len(destination_raw_blob_path):]
                  for file in file_names]

    for file in file_names:
        raw_blob = bucket.blob(destination_raw_blob_path + str(file))
        raw_blob.download_to_filename(file)
    logging.info('Finish the download on raw data: SUCCESS')

    # 2. make some transformations (transform)
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
            try:
                current_pdf = pd.read_csv(directory + '/' + each_filename)
                pdf = pdf.append(current_pdf).reset_index(drop=True)
            except BaseException:
                continue

    drop_unnecessary_cols = [col for col in pdf.columns if 'Unnamed' in col]
    pdf.drop(drop_unnecessary_cols, axis=1, inplace=True)
    pdf.drop_duplicates(inplace=True)
    logging.info('Finish making the transformations on raw data: SUCCESS')

    # 3. upload the transformed data into trusted layer in google storage and wandb (load)
    # 3.1 google storage
    trusted_blob = bucket.blob(destination_trusted_blob_path)
    trusted_blob.upload_from_string(
        current_pdf.to_csv(
            name_set + '.csv',
            index=False))
    logging.info('Uploaded train_set.csv to google storage: SUCCESS')

    # 3.2 wandb
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='upload trusted data (transformed data)')
    logging.info('Creating run for risk assessment: SUCCESS')

    artifact = wandb.Artifact(
        name=name_set,
        type='dataset',
        description='Raw dataset transformed with some necessary things to start DS pipeline')

    pdf.to_csv(name_set, index=False)
    artifact.add_file(name_set + '.csv')
    run.log_artifact(artifact)
    logging.info(f'Uploaded {name_set} to wandb: SUCCESS')

    # 4. Exclude local unnecessary files files
    for each_filename in filenames:
        if each_filename.endswith('.csv'):
            os.remove(each_filename)


if __name__ == "__main__":
    logging.info('About to start executing of the transform function')

    dataset_names = ['train_set', 'test_set']
    dataset_folders = ['train/', 'test/']
    for folder in dataset_folders:
        transform_raw_data(
            BUCKET_NAME,
            DESTINATION_RAW_BLOB_PATH + folder,
            COMPONENT_CURRENT_DIRECTORY,
            DESTINATION_TRUSTED_BLOB_PATH + folder,
            dataset_names
        )

    logging.info('Done executing the transform function')
