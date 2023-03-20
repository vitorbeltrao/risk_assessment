'''
Script to take the raw data from the data lake and do some 
transformations to feed the trusted layer of the data lake

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

# # config
# BUCKET_NAME = sys.argv[1]
# DESTINATION_TRAIN_BLOB_PATH = sys.argv[2]
# RAW_DATA_PATH = sys.argv[3]
# key code for managing the entire infrastructure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'


def transform_raw_data(
        bucket_name: str,
        destination_blob_path: str) -> None:
    '''Function that takes all datasets that are in the raw layer 
    of the data lake and applies two transformations: union of 
    datasets and drop duplicate data
    '''
    # 1. data download (extract)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_names = list(storage_client.list_blobs(bucket_name, prefix=destination_blob_path))
    file_names = [file.name[len(destination_blob_path):] for file in file_names]

    for file in file_names:
        blob = bucket.blob(destination_blob_path + str(file))
        blob.download_to_filename(file)

    # 2. make some transformations (transform)
    train_data_directory = ['C:/Users/4YouSee/Desktop/personal_work/risk_assessment/components/02_upload_trusted_data']
    train_df = pd.DataFrame(columns=['corporation','lastmonth_activity', 'lastyear_activity', 
                                     'number_of_employees', 'exited'])
    for directory in train_data_directory:
        filenames = os.listdir(directory)
        for each_filename in filenames:
            # some files in this directory are not .csv
            try:
                current_train_df = pd.read_csv(directory + '/' + each_filename)
                train_df = train_df.append(current_train_df).reset_index(drop=True)
            except:
                continue

    drop_unnecessary_cols = [col for col in train_df.columns if 'Unnamed' in col] 
    train_df.drop(drop_unnecessary_cols, axis=1, inplace=True)
    train_df.drop_duplicates(inplace=True)
    train_df.to_csv('train_df.csv', index=False)


if __name__ == "__main__":
    logging.info('About to start executing of the functions')

    transform_raw_data('risk_assessment_storage', 'raw/train_data/')
    a = pd.read_csv('train_df.csv')
    print(a)

    logging.info('Done executing the functions')


