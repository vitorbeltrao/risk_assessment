'''
Script to create a bucket on google storage instance

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import os
from decouple import config
from google.cloud import storage

# config
# key code for managing the entire infrastructure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config('JSON_KEY')


def create_bucket(
        bucket_name: str, 
        storage_class:str='STANDARD', 
        location:str='us-east1') -> str: 
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
    bucket.storage_class = storage_class
    
    bucket = storage_client.create_bucket(bucket, location=location)
    # for dual-location buckets add data_locations=[region_1, region_2]
    
    return f'Bucket {bucket.name} successfully created.'