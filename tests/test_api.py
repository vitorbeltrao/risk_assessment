'''
Unit test of ml_api.py API module with pytest

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import json
import logging
from fastapi.testclient import TestClient
from model_drift_check.model_drift_check import final_model_drift_verify
from ml_api import app

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# create a test client instance
client = TestClient(app)


def test_get():
    '''Test welcome message for get at root'''
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == 'Welcome to our model API'


def test_inference_class1():
    '''Test model inference output for class 1'''
    sample = {
        "corporation": "nciw",
        "lastmonth_activity": 45, 
        "lastyear_activity": 0,
        "number_of_employees": 99
    }

    data = json.dumps(sample)

    response = client.post('/risk_assessment_prediction', data=data)

    # test response and output
    assert response.status_code == 200
    assert response.json() == 'The person is at risk of leaving the company'


def test_inference_class0():
    '''Test model inference output for class 0'''
    sample =  {
        "corporation": "lsid",
        "lastmonth_activity": 36, 
        "lastyear_activity": 234,
        "number_of_employees": 541
    }

    data = json.dumps(sample)

    response = client.post('/risk_assessment_prediction', data=data)

    # test response and output
    assert response.status_code == 200
    assert response.json() == 'The person has no risk of leaving the company'
