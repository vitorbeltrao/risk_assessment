'''
Script to upload the original files 
to the raw layer of the data lake

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import logging
import pandas as pd
from google.cloud import storage

def upload_to_storage(
        bucket_name: str,
        data: pd.DataFrame,
        destination_blob_path: str,
        download_file: str) -> None:
    
    '''Function that uploads a dataframe into a google storage bucket

    :param bucket_name: (str)
    Name of the respective bucket

    :param data: (dataframe)
    Dataframe you want to upload

    :param destination_blob_path: (str)
    destination path of the file you want to upload to the google storage bucket

    :param download_file: (str)
    Name of the file you want to download from your google storage bucket
    '''
