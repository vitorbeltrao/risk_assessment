'''
This file is for creating our inference api with fastapi

Author: Vitor Abdo
Date: March/2023
'''

# Import necessary packages
import json
import logging
import pickle
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

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


# get mlflow model pkl from prod_deployment_path folder
model_path = os.path.join('prod_deployment_path', 'model.pkl')
sk_pipe = pickle.load(open(model_path, 'rb'))
logging.info('Get prod mlflow model: SUCCESS')


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


if __name__ == '__main__':
    pass