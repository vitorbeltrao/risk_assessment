'''
DAG creation in airflow

Author: Vitor Abdo
Date: April/2023
'''

# import necessary packages
import logging
from datetime import datetime

from airflow import DAG 
from airflow.operators.python import PythonOperator # operator of the task that we are going to use, in this case python
from airflow.operators.bash import BashOperator # operator to run a bash command

from components.upload_raw_data import upload_raw_data
from components.upload_trusted_data import upload_trusted_data
from model_data_diagnostics import data_drift_check
from model_data_diagnostics import model_drift_check

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# create the dag
with DAG('diagnostics', start_date=datetime(2023,4,8),
          schedule_interval='0 3 * * *', catchup=False) as dag:

    # 1st task -> Upload data to the raw folder of the data lake    
    upload_raw = PythonOperator(
        task_id = 'upload_raw_data',
        python_callable = upload_raw_data # function that feeds this task
    )

    # 2nd task -> Upload data to the trusted folder of the data lake  
    upload_trusted = PythonOperator(
        task_id = 'upload_trusted_data',
        python_callable = upload_trusted_data # function that feeds this task
    )

    # 3rd task -> Run the data diagnostics functions in parallel
    data_drift = PythonOperator(
        task_id='data_drift_check',
        python_callable=data_drift_check
    )

    model_drift = PythonOperator(
        task_id='model_drift_check',
        python_callable=model_drift_check
    )

    # dependencies of the tasks
    [upload_raw, upload_trusted] >> [data_drift, model_drift]

    # 4th task -> Verify the result of the "model_drift_check" function and run the command "mlflow run ." if True
    mlflow_run = BashOperator(
        task_id='mlflow_run',
        bash_command='mlflow run .',
        trigger_rule='all_done',
        dag=dag
    )

    # dependency of the task
    model_drift >> mlflow_run
  