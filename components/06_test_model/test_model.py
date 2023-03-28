'''
This file is for testing the final model with the "prod" tag in the test data

Author: Vitor Abdo
Date: March/2023
'''

# Import necessary packages
import pickle
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')

# config
FINAL_MODEL = sys.argv[1]
TEST_SET = sys.argv[2]
LABEL_COLUMN = sys.argv[3]
SCORE_MEASURE_SLICE = sys.argv[4]


def evaluate_model(
        final_model: pickle,
        test_set: str,
        label_column: str,
        score_measure_slice: str,) -> None:
    '''Function to test the model listed for production on the test dataset

    :param final_model: (pickle)
    Pickle file with all saved model pipeline

    :param test_set: (str)
    Path to the wandb leading to the test dataset

    :param label_column: (str)
    Column name of the dataset to be trained that will be the label

    :param score_measure_slice: (str)
    What performance metric do you want to measure your results on the sliced data
    '''
    # start a new run at wandb
    run = wandb.init(
        project='risk_assessment',
        entity='vitorabdo',
        job_type='test_model')

    # download mlflow model
    model_local_path = run.use_artifact(final_model, type='pickle').download()
    logging.info('Downloaded prod mlflow model: SUCCESS')

    # download test dataset
    test_data = run.use_artifact(test_set, type='dataset').file()
    logging.info('Downloaded test dataset artifact: SUCCESS')

    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop([label_column], axis=1)
    y_test = test_data[label_column]

    # making inference on test set
    logging.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    # scoring the results in a txt file
    logging.info("Scoring...")
    metrics_filename = open('metrics_output', 'w')
    sys.stdout = metrics_filename
    print(classification_report(y_test, y_pred))
    print("AUC: {:.4f}\n".format(roc_auc_score(y_test, y_pred)))
    metrics_filename.close()

    # plot confusion matrix
    conf_matrix, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'),
                annot=True, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    conf_matrix.tight_layout()

    # plot roc_auc
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    dt_roc_model = RocCurveDisplay.from_estimator(
        sk_pipe, X_test, y_test, ax=ax, alpha=0.8)
    dt_roc_model.plot(ax=ax, alpha=0.8)
    plt.tight_layout()

    # upload the plots in wandb
    run.log(
        {'confusion_matrix': wandb.Image(conf_matrix)},
        {'roc_curve': wandb.Image(dt_roc_model)})

    # print metrics with our data sliced in a txt file
    slice_filename = open('slice_output', 'w')
    sys.stdout = slice_filename

    # sliced data for categorical values
    for columns in X_test.columns:
        print(f'\n{score_measure_slice} on {columns} slices:')
        slice_options = X_test[columns].unique().tolist()
        for option in slice_options:
            row_slice = X_test[columns] == option
            print(f'{option}', score_measure_slice(
                y_test[row_slice], sk_pipe.predict(X_test[row_slice])))

    slice_filename.close()


if __name__ == "__main__":
    logging.info('About to start executing the test_model function')
    evaluate_model(FINAL_MODEL, TEST_SET, LABEL_COLUMN, SCORE_MEASURE_SLICE)
    logging.info('Done executing the test_model function')
