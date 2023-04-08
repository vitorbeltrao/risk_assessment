'''
This file is for doing some checks on model drift.
If the result shows model drift, then the system 
sends an email to the owner.

Author: Vitor Abdo
Date: March/2023
'''

# Import necessary packages
import os
import pickle
import logging
import pandas as pd
import numpy as np
import wandb
import smtplib
from decouple import config
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# start a new run at wandb
run = wandb.init(
    project='risk_assessment',
    entity='vitorabdo',
    job_type='check_model_drift')

# config email informations: Set sender and recipient information
FROM = config('FROM')
TO = config('TO')
PASS = config('PASS')

def raw_comparison_test(hist_metrics: str, newf1score: int) -> bool:
    '''raw comparison: we simply check whether current 
    performance is worse than all previous scores. If 
    the current performance score is worse than all 
    previous scores, then we say that model drift 
    has occurred - according to the raw comparison test

    :param hist_metrics: (str)
    Path to the wandb leading to the historical metrics csv

    :param newf1score: (int)
    F1 score value to be compared with historical metrics set

    :return: (bool)
    Returns bool showing whether or not the model drift occurred. 
    If true: model drift occurred
    '''
    # Read historical metrics dataset
    previousscores = pd.read_csv(hist_metrics)

    # raw comparison
    raw_test = newf1score < previousscores['metric_f1score'].values.min()
    return raw_test


def parametric_significance_test(hist_metrics: str, newf1score: int) -> bool:
    '''This test will check the standard deviation of all previous scores. 
    Then, it will conclude that a new model has worse performance than 
    previous models if the new model score is more than two standard 
    deviations lower than the mean of all the previous scores

    :param hist_metrics: (str)
    Path to the wandb leading to the historical metrics csv

    :param newf1score: (int)
    F1 score value to be compared with historical metrics set

    :return: (bool)
    Returns bool showing whether or not the model drift occurred. 
    If true: model drift occurred
    '''
    # Read historical metrics dataset
    previousscores = pd.read_csv(hist_metrics)

    # parametric comparison
    parametric_test = newf1score < np.mean(previousscores['metric_f1score']) - (2 * np.std(previousscores['metric_f1score']))
    return parametric_test


def non_parametric_outlier_test(hist_metrics: str, newf1score: int) -> bool:
    '''Instead of the standard deviation, this test uses the interquartile range: 
    the difference between the 75th percentile and the 25th percentile. A model 
    score is regarded as an extreme value if it is either:

    1. more than 1.5 interquartile ranges above the 75th percentile (a high outlier)
    2. more than 1.5 interquartile ranges below the 25th percentile (a low outlier)

    If a model score is worse than previous scores to an extent that it's an outlier 
    (either a high or low outlier), then the non-parametric outlier test concludes 
    that model drift has occurred

    :param hist_metrics: (str)
    Path to the wandb leading to the historical metrics csv

    :param newf1score: (int)
    F1 score value to be compared with historical metrics set

    :return: (bool)
    Returns bool showing whether or not the model drift occurred. 
    If true: model drift occurred
    '''
    # Read historical metrics dataset
    previousscores = pd.read_csv(hist_metrics)

    # non parametric comparison
    iqr = np.quantile(previousscores['metric_f1score'], 0.75) - np.quantile(previousscores['metric_f1score'], 0.25)
    non_parametric_test = newf1score < np.quantile(previousscores['metric_f1score'], 0.25) - iqr * 1.5
    return non_parametric_test


def final_model_drift_verify(hist_metrics: str, newf1score: int) -> bool:
    '''Test that checks the three methods:
    raw comparison test, parametric_significance_test
    and non parametric outlier test. We will have a model 
    drift if at least 2 tests show that there was a model 
    drift. Remembering, the True result is a sign that the 
    test accused the model drift

    :param hist_metrics: (str)
    Path to the wandb leading to the historical metrics csv

    :param newf1score: (int)
    F1 score value to be compared with historical metrics set

    :return: (bool)
    Returns bool showing whether or not the model drift occurred. 
    If true: model drift occurred
    '''
    # final comparison
    first_test = raw_comparison_test(hist_metrics, newf1score)
    second_test = parametric_significance_test(hist_metrics, newf1score)
    third_test = non_parametric_outlier_test(hist_metrics, newf1score)

    if first_test is True and second_test is True:
        print('True')
        return True
    elif first_test is True and third_test is True:
        print('True')
        return True
    elif second_test is True and third_test is True:
        print('True')
        return True
    else:
        print('False')
        return False


if __name__ == '__main__':
    logging.info('About to start the model drift check\n')

    # download historical metrics dataset
    previousscores = run.use_artifact('vitorabdo/risk_assessment/historical_metrics:latest', 
                                      type='dataset').file()
    
    # get mlflow model
    model_path = os.path.join('../prod_deployment_path', 'model.pkl')
    sk_pipe = pickle.load(open(model_path, 'rb'))
    
    # download test dataset
    test_data = run.use_artifact('vitorabdo/risk_assessment/test_set.csv:latest', 
                                 type='dataset').file()
    wandb.finish()
    
    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(['exited'], axis=1)
    y_test = test_data['exited']

    # making inference on test set
    y_pred = sk_pipe.predict(X_test)
    new_f1score = f1_score(y_test, y_pred)

    # Test to check if there was model drift
    result = final_model_drift_verify(previousscores, new_f1score)
    if result is False:
        logging.info('We dont have model drift: SUCCESS')
    else:
        logging.info('We have model drift: RETRAIN AND RE-DEPLOY THE MODEL')

        # create message body and email subject
        subject = 'We have model drift: RETRAIN AND RE-DEPLOY THE MODEL'
        body = 'Check your system, as we have a possible model drift occurring.'

        # connect to the SMTP server and login
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM, PASS)

        # create the message and send the email
        message = f'Subject: {subject}\n\n{body}'
        server.sendmail(FROM, TO, message)
        server.quit()

    logging.info('Done executing the model drift check')
