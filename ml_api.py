'''
This file is for creating our inference api with fastapi

Author: Vitor Abdo
Date: March/2023
'''

# Import necessary packages
import json
import logging
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import wandb
import subprocess
import re
import diagnostics

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


# Creating a Fastapi object
app = FastAPI()

class ModelInput(BaseModel):
    '''identifying the type of our model features'''
    corporation: str
    lastmonth_activity: int
    lastyear_activity: int
    number_of_employees: str
  
    class Config:
        schema_extra = {
            "example": {
            "corporation": "nciw",
            "lastmonth_activity": 45, 
            "lastyear_activity": 0,
            "number_of_employees": 99
            }
        }


# loading the saved model
# start a new run at wandb
run = wandb.init(
    project='risk_assessment',
    entity='vitorabdo',
    job_type='get_mlflow_model')

# download mlflow model
model_local_path = run.use_artifact(
    'vitorabdo/risk_assessment/final_model_pipe:prod',
    type='pickle').download()
sk_pipe = mlflow.sklearn.load_model(model_local_path)
wandb.finish()
logging.info('Downloaded prod mlflow model: SUCCESS')


@app.get('/')
def greetings():
    '''get method to to greet a user'''
    return 'Welcome to our model API'


@app.post('/risk_assessment_prediction')
def income_pred(input_parameters: ModelInput):
    '''post method to our inference'''

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    input_df = pd.DataFrame(
        input_dictionary,
        columns=sk_pipe.named_steps['preprocessor'].transformers_[0][2],
        index=[0])

    prediction = sk_pipe.predict(input_df)

    if prediction[0] == 0:
        return 'The person has no risk of leaving the company'

    return 'The person is at risk of leaving the company'


@app.get("/scoring")
def score():
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model
    Returns:
        str: model f1 score
    """
    output = subprocess.run(['python', 'components/06_test_model/test_model.py'],
                            capture_output=True).stdout
    output = re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    return output


@app.get("/summarystats")
def stats():
    """
    Summary statistics endpoint that calls dataframe summary
    function from diagnostics.py
    Returns:
        json: summary statistics
    """
    return diagnostics.dataframe_summary()


@app.get("/diagnostics")
def diag():
    """
    Diagnostics endpoint thats calls missing_percentage, execution_time,
    and outdated_package_list from diagnostics.py
    Returns:
        dict: missing percentage, execution time and outdated packages
    """
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    ret = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return ret



if __name__ == '__main__':
    pass