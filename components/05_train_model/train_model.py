'''
This .py file is for training, saving the best model and
get the feature importance for model

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import timeit
import logging
import os
import tempfile
import math
import mlflow
import sys
import json
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# config
TRAIN_SET = sys.argv[1]
LABEL_COLUMN = sys.argv[2]
CV = int(sys.argv[3])
SCORING = sys.argv[4]
RF_CONFIG = sys.argv[5]


def get_inference_pipeline(X: pd.DataFrame) -> Pipeline:
    '''function that creates the entire inference pipeline

    :param X: (DataFrame)
    Independent dataset to feed the ML model

    :return: (Pipeline)
    The pipeline that made the final model
    '''
    # 1. make pipelines to do the necessary transformations
    # 1.1 divide the qualitative and quantitative features
    quantitative_columns = selector(dtype_exclude=['object'])
    quantitative_columns = quantitative_columns(X)

    # 1.2 apply the respective transformations with columntransformer method
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), quantitative_columns)],
        remainder='drop')

    processed_features = quantitative_columns

    # instantiate the final model
    final_model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('dt', DecisionTreeClassifier(random_state=42))]
    )
    return final_model, processed_features


def plot_feature_importance(pipe, feat_names) -> plt.figure:
    '''Function to generate the graph of the
    most important variables for the model

    :param pipe: (Pipeline)
    The pipeline that made the final model

    :param feat_names: (list)
    List with the name of the variables used in your model

    :return: (figure)
    Returns the figure with the graph of the most
    important variables for the model
    '''
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["dt"].feature_importances_[: len(feat_names)]

    # plot the figure
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def train_model(
        train_set: str,
        label_column: str,
        cv: int,
        scoring: str,
        rf_config: dict) -> None:
    '''Function to train the model, tune the hyperparameters
    and save the best final model

    :param train_set: (str)
    Path to the wandb leading to the training dataset

    :param label_column: (str)
    Column name of the dataset to be trained that will be the label

    :param cv: (int)
    Determines the cross-validation split strategy

    :param scoring: (str)
    Strategy to evaluate the performance of the model of
    cross-validation in the validation set

    :param rf_config: (dict)
    Dict with the values of the hyperparameters of the adopted model
    '''
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='train_data')
    artifact = run.use_artifact(train_set, type='dataset')
    filepath = artifact.file()
    logging.info('Downloaded cleaned data artifact: SUCCESS')

    # Get the decision tree configuration and update W&B
    try:
        with open({rf_config}) as fp:
            rf_config = json.load(fp)
        run.config.update(rf_config)
    except BaseException:
        rf_config = {}

    # select only the features that we are going to use
    df_clean = pd.read_csv(filepath)
    X = df_clean.drop([label_column], axis=1)
    y = df_clean[label_column]
    logging.info(f"Numbers of unique incomes: {y.value_counts()}")

    # training the model
    logging.info('Preparing sklearn pipeline')
    sk_pipe, processed_features = get_inference_pipeline(X)

    # hyperparameter interval to be trained and tested
    logging.info('Fitting...')
    param_grid = rf_config

    grid_search = GridSearchCV(
        sk_pipe,
        param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True)
    grid_search.fit(X, y)

    # instantiate best model
    final_model = grid_search.best_estimator_

    # scoring
    logging.info('Scoring...')
    cvres = grid_search.cv_results_

    cvres = [(mean_test_score, mean_train_score) for mean_test_score,
              mean_train_score in sorted(zip(cvres['mean_test_score'],
                                             cvres['mean_train_score']),
                                         reverse=True) if (math.isnan(mean_test_score) != True)]

    logging.info(
        f"The mean val score and mean train score of {scoring} is, respectively: {cvres[0]}")

    # exporting the model: save model package in the MLFlow sklearn format
    logging.info('Exporting model')

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, 'model_export')

    mlflow.sklearn.save_model(
        final_model,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    # upload the model artifact into wandb
    artifact = wandb.Artifact(
        name='final_model_pipe',
        type='pickle',
        description='Final model pipeline after training, exported in the correct format for making inferences')

    artifact.add_dir(export_path)
    run.log_artifact(artifact)
    artifact.wait()
    logging.info('Artifact Uploaded: SUCCESS')

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(final_model, processed_features)

    # lets save and upload all metrics to wandb
    run.summary['Train_score'] = cvres[0][1]
    run.summary['Val_score'] = cvres[0][0]

    run.log(
        {
            'feature_importance': wandb.Image(fig_feat_imp)
        }
    )


if __name__ == "__main__":
    logging.info('About to start executing the train_model function')
    starttime = timeit.default_timer() 

    train_model(TRAIN_SET, LABEL_COLUMN, CV, SCORING, RF_CONFIG)
    
    timing = timeit.default_timer() - starttime
    logging.info('The execution time of this step was:', timing)
    logging.info('Done executing the train_model function')
