'''
Script to create a bucket on google storage instance

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import os
from google.cloud import storage

# config
# key code for managing the entire infrastructure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'


def create_bucket(
        bucket_name: str,
        storage_class: str = 'STANDARD',
        location: str = 'us-east1') -> str:
    '''function that creates the bucket

    :param bucket_name: (str)
    Name of the respective bucket

    :param storage_class: (str)
    Choose a storage class for your data:
    Standard, Nearline, Coldline, Archive

    :param location: (str)
    Choose where to store your data
    '''
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    if bucket.exists() is False:
        bucket.storage_class = storage_class
        bucket = storage_client.create_bucket(bucket, location=location)
        # for dual-location buckets add data_locations=[region_1, region_2]

        return print(f'Bucket {bucket.name} successfully created.')
    return print(f'Bucket {bucket.name} already exists.')


if __name__ == "__main__":
    create_bucket('risk_assessment_storage')
