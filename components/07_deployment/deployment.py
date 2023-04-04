'''
Function that will copy the files in production to organize them

Author: Vitor Abdo
Date: April/2023
'''

# Import necessary packages
import os
import pickle
import logging
import sys
import wandb
import shutil

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# # config
# PROD_DEPLOYMENT_PATH = sys.argv[1]
# FINAL_MODEL = sys.argv[2]
# LATEST_SCORES_PATH = sys.argv[3]
# INGESTED_FILE_PATH = sys.argv[4]

PROD_DEPLOYMENT_PATH = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/prod_deployment_path'
FINAL_MODEL = 'vitorabdo/risk_assessment/final_model_pipe:prod'
LATEST_SCORES_PATH = '../06_test_model/'
INGESTED_FILE_PATH = 'C:/Users/4YouSee/Desktop/personal_work/risk_assessment/components/01_upload_raw_data/ingested_files.txt'


def deploy_model(prod_deployment_path: str, 
                 final_model: pickle,
                 latest_scores_path: str,
                 ingested_files_path: str):
    '''
    function for deployment, copy the latest pickle file, the latestscore.txt value, 
    and the ingestedfiles.txt file into the deployment directory
    '''
    logging.info(
        f'Model deployment. Model, its latest score, and the list of files used for training are saved in '
        f'{prod_deployment_path}')

    # # obtaining the necessary files
    # # 1. final model pickle
    # run = wandb.init(
    #     project='risk_assessment',
    #     entity='vitorabdo',
    #     job_type='deployment')
    # model_local_path = run.use_artifact(final_model, type='pickle').download()
    # print(model_local_path)
    # logging.info('Downloaded prod mlflow model: SUCCESS')

    # 2. latest scores of the model
    score_path = latest_scores_path
    print(score_path)

    # 3. ingested files
    ingested_path = ingested_files_path
    print(ingested_path)

    sources = [score_path, ingested_path]
    print(sources)

    # create production deployment folder
    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    shutil.copy(
        os.path.join(
            score_path,
            'actual_metrics_output'),
        prod_deployment_path)

    # for source, target in zip(sources, targets):
    #     os.system(f'cp {source} {target}')


if __name__ == '__main__':
    logging.info('About to start executing the deployment function')
    deploy_model(PROD_DEPLOYMENT_PATH, FINAL_MODEL, LATEST_SCORES_PATH, INGESTED_FILE_PATH)
    logging.info('Done executing the deployment function')