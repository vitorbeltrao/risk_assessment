'''
This is the main system file that runs all necessary
components to run the machine learning pipeline

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import os
import subprocess

# Steps to execute
_steps = [
    '01_upload_raw_data',
    '02_upload_trusted_data',
    '03_basic_clean',
    '04_data_check',
    '05_train_model',
    '06_test_model',
    '07_deployment'
]

def run_step(step):
    '''Runs the specified step using MLflow'''
    print(f"Running step: {step}")
    component_dir = f"https://github.com/vitorbeltrao/risk_assessment/components/{step}"
    subprocess.call(f"mlflow run {component_dir} -P steps={step}", shell=True)

def run_pipeline(steps):
    '''Runs the pipeline with the specified steps'''
    for step in steps:
        run_step(step)

if __name__ == "__main__":
    # Get the steps to execute from the environment variable, or execute all steps by default
    steps_par = os.getenv('STEPS', 'all')
    active_steps = _steps if steps_par.lower() == 'all' else steps_par.split(',')

    # Set the wandb environment variables
    os.environ['WANDB_PROJECT'] = "risk_assessment"
    os.environ['WANDB_RUN_GROUP'] = "pipeline_execution"

    # Run the pipeline with the specified steps
    run_pipeline(active_steps)
