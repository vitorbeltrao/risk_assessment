'''
Script to upload the original files
to the raw layer of the data lake

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import timeit
import os
import sys
import logging
import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
BUCKET_NAME = sys.argv[1]
DESTINATION_TRAIN_BLOB_PATH = sys.argv[2]
DESTINATION_TEST_BLOB_PATH = sys.argv[3]
TRAIN_DATA_FOLDER_PATH = sys.argv[4]
TEST_DATA_FOLDER_PATH = sys.argv[5]
# key code for managing the entire infrastructure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'


def import_data(file_path: str) -> pd.DataFrame:
    '''Load dataset as a pandas dataframe for the csv found at the path

    :param file_path: (str)
    A path to the csv

    :return: (dataframe)
    Pandas dataframe
    '''
    try:
        raw_df = pd.read_csv(file_path)
        logging.info('Execution of import_data: SUCCESS')
        return raw_df

    except FileNotFoundError:
        logging.error("Execution of import_data: The file wasn't found")
        return None


def upload_to_storage(
        bucket_name: str,
        data: pd.DataFrame,
        destination_blob_path: str) -> None:
    '''Function that uploads a dataframe into a google storage bucket

    :param bucket_name: (str)
    Name of the respective bucket

    :param data: (dataframe)
    Dataframe you want to upload

    :param destination_blob_path: (str)
    destination path of the file you want to upload to the google storage bucket
    '''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_string(data.to_csv())

    return logging.info(
        'Loading the NEW CSV FILE into the bucket: SUCCESS\n')


if __name__ == "__main__":
    logging.info('About to start executing of the functions\n')
    starttime = timeit.default_timer()

    # 1. upload train data to raw/train_data folder in the bucket
    train_data_directory = [TRAIN_DATA_FOLDER_PATH]
    ingested_files = open('ingested_files', 'w')
    sys.stdout = ingested_files
    for directory in train_data_directory:
        filenames = os.listdir(directory)
        for each_filename in filenames:
            current_train_df = import_data(directory + '/' + each_filename)
            upload_to_storage(
                BUCKET_NAME,
                current_train_df,
                DESTINATION_TRAIN_BLOB_PATH +
                each_filename)
            
            # printing ingested files to a txt file
            print(each_filename)
    ingested_files.close()

    # 2. upload test data to raw/test_data folder in the bucket
    test_data_directory = [TEST_DATA_FOLDER_PATH]
    for directory in test_data_directory:
        filenames = os.listdir(directory)
        for each_filename in filenames:
            current_test_df = import_data(directory + '/' + each_filename)
            upload_to_storage(
                BUCKET_NAME,
                current_test_df,
                DESTINATION_TEST_BLOB_PATH +
                each_filename)
            
    timing = timeit.default_timer() - starttime
    logging.info(f'The execution time of this step was:{timing}')
    logging.info('Done executing the functions')
