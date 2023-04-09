'''
This file is to verify and monitor the data drift, 
at the end it generates a report on the data drift 
and one on the stability of the data.

Author: Vitor Abdo
Date: April/2023
'''

# import necessary packages
import logging
import io
import os
import pandas as pd
import wandb
from google.cloud import storage
from decouple import config


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests import *

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# config
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'
BUCKET_NAME = config('BUCKET_NAME')
FILE_PATH = config('FILE_PATH')
REF_DATASET = config('REF_DATASET')


def read_gcs_csv_to_dataframe(bucket_name: str, file_path: str) -> pd.DataFrame:
    '''Reads a CSV file from Google Cloud Storage and returns it as a Pandas DataFrame

    :param bucket_name: (str)
    The name of the bucket containing the file

    :param file_path: (str)
    The path to the CSV file within the bucket

    :return: (pandas.DataFrame)
    A DataFrame containing the contents of the CSV file.
    '''

    # Create a client to access Google Cloud Storage
    storage_client = storage.Client()

    # Get the bucket that contains the file
    bucket = storage_client.get_bucket(bucket_name)

    # Get a blob object that represents the CSV file
    blob = bucket.blob(file_path)

    # Download the file as a string
    csv_data = blob.download_as_string()

    # Parse the CSV data into a Pandas DataFrame
    historical_df = pd.read_csv(io.BytesIO(csv_data))

    return historical_df


def download_reference_dataset(ref_dataset: str) -> pd.DataFrame:
    '''Function that downloads the reference dataset to monitor data drift

    :param ref_dataset: (str)
    Path to the wandb leading to the reference cleaned dataset

    :return: (pandas.DataFrame)
    A DataFrame containing the downloaded table.
    '''

    # start a new run at wandb and read the dataset as a pandas df
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='check_data_drift')

    artifact = run.use_artifact(ref_dataset, type='dataset')
    filepath = artifact.file()
    wandb.finish()
    df_reference = pd.read_csv(filepath)

    return df_reference


if __name__ == '__main__':
    logging.info('About to start the data drift check\n')

    # download datasets
    current = read_gcs_csv_to_dataframe(BUCKET_NAME, FILE_PATH)
    logging.info('Historical dataset downloaded: SUCCESS')

    reference = download_reference_dataset(REF_DATASET)
    logging.info('Reference dataset downloaded: SUCCESS')

    # generate evidently data drift report
    report = Report(metrics=[
    DataDriftPreset(), 
    ])

    report.run(reference_data=reference, current_data=current)
    report.save_html('model_data_diagnostics/data_drift_report.html')
    logging.info('Generate data drift report: SUCCESS')

    # generate evidently data stability report
    tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=reference, current_data=current)
    tests.save_html('model_data_diagnostics/data_stability_report.html')
    logging.info('Generate data stability report: SUCCESS')

    logging.info('Done executing the data drift check')