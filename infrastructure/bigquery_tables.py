'''
Script that moves csv tables from cloud storage to bigquery

Author: Vitor Abdo
Date: April/2023
'''

# import necessary packages
import os
import logging
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# config
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/risk-assessment-380822-38a40f93abec.json'
URI = 'gs://risk_assessment_storage/trusted/train_data/train_set.csv'


def create_bigquery_table(uri: str) -> None:
    '''Function that moves data from google storage to the warehouse

    :param uri: (str)
    Path of the table in cloud storage that you want to move to a table in bigquery
    '''
    # Construct a BigQuery client object
    client = bigquery.Client()

    # Create the dataset
    dataset_id = 'historical_train_data'
    dataset_ref = client.dataset(dataset_id)

    # Create schema
    schema=[
        bigquery.SchemaField('ID', 'INTEGER'),
        bigquery.SchemaField('corporation', 'STRING'),
        bigquery.SchemaField('lastmonth_activity', 'INTEGER'),
        bigquery.SchemaField('lastyear_activity', 'INTEGER'),
        bigquery.SchemaField('number_of_employees', 'INTEGER'),
        bigquery.SchemaField('exited', 'INTEGER')]

    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = 'US'
        client.create_dataset(dataset)
        logging.info('Dataset created: SUCCESS')
    except:
        logging.info('Dataset already exists')

    # Create the table if it does not exist
    table_id = 'train_set'
    table_ref = dataset_ref.table(table_id)

    try:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        logging.info('Table created: SUCCESS')
    except:
        logging.info('Table already exists')

    # Load the CSV to a temporary table
    temp_table_id = 'train_set_temp'
    temp_table_ref = dataset_ref.table(temp_table_id)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
    )
    uri = uri

    load_job = client.load_table_from_uri(
        uri, temp_table_ref, job_config=job_config
    )
    load_job.result()

    temp_table = client.get_table(temp_table_ref)
    # Merge the data into the main table
    merge_sql = f'''
    MERGE `{dataset_id}.{table_id}` t
    USING `{dataset_id}.{temp_table_id}` s
    ON t.ID = s.ID
    WHEN NOT MATCHED THEN
    INSERT (
        ID,
        corporation,
        lastmonth_activity,
        lastyear_activity,
        number_of_employees,
        exited
    )
    VALUES (
        s.ID,
        s.corporation,
        s.lastmonth_activity,
        s.lastyear_activity,
        s.number_of_employees,
        s.exited
    )
    '''

    query_job = client.query(merge_sql)
    query_job.result()
    logging.info(f'Loaded {temp_table.num_rows} rows in {temp_table_id} : SUCCESS')

    # Delete the temporary table
    client.delete_table(temp_table)
    logging.info(f'Deleted {temp_table_id}: SUCCESS')


if __name__ == '__main__':
    logging.info('About to start the creation of bigquery tables')
    create_bigquery_table(URI)
    logging.info('Done executing the creation of bigquery tables')
