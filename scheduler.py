'''
Scheduler creation to orchestrate the system

Author: Vitor Abdo
Date: April/2023
'''

# import necessary packages
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

def run_mlflow():
    subprocess.run(['mlflow', 'run', '.', '-P', 'steps=upload_raw_data,upload_trusted_data'])

def run_diagnostics():
    subprocess.run('python model_data_diagnostics/data_drift_check.py')
    subprocess.run('python model_data_diagnostics/model_drift_check.py')

def run_mlflow_if_model_drift():
    with open('model_data_diagnostics/model_drift_check_result.txt', 'r') as f:
        result = f.read().strip()
        if result == 'True':
            subprocess.run(['mlflow', 'run', '.'])

if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(run_mlflow, 'cron', hour=3)
    scheduler.add_job(run_diagnostics, 'cron', hour=4)
    scheduler.add_job(run_mlflow_if_model_drift, 'cron', hour=5)
    scheduler.start()
