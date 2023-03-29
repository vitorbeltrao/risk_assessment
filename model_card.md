# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The Dynamic Risk Assessment System project is based on a machine learning model that predicts whether a customer will stop using the company's services. In this case, we created a model that will make this prediction, so that the company can take some action to prevent the customer from leaving the company, and thus avoid losses.

We used a dataset provided by the company to train the model. The independent variables and the dependent variable (target) were provided so that this task could be carried out. We train and validate the model with a part of the data set (training data) and evaluate the performance of the model in production with the test data. 

We used a [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), class from scikit-learn in version 1.2.1. To arrive at the final version, we used the aforementioned model and tuned the hyperparameters with the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class. The optimal hyperparameters were:

* max_depth: 3 

The rest of the hyperparameters were kept by default.

Model is saved as a pickle file stored in the [wandb](https://wandb.ai/site). To use the latest model available in production, you must download it via wandb, like this: `model_local_path = run.use_artifact('[your_wandb_username]/[your_wandb_projectname]/[your_wandb_filename]:prod', type='pickle').download()`

Model training and validation was done in component [05_train_model](https://github.com/vitorbeltrao/risk_assessment/tree/main/components/05_train_model).

Model testing was done in component [06_test_model](https://github.com/vitorbeltrao/risk_assessment/tree/main/components/06_test_model).

## Intended Use

Currently, companies must, in addition to seeking new customers, keep current customers to avoid losses. The idea of this project is precisely to predict whether a customer will stop using the company's services so that it can take the necessary actions to prevent this. The usage is meant for students, academics or research purpose.

## Training Data

The dataset was provided by the company. For more information, contact the author of the project. The means of communication are in the [README](https://github.com/vitorbeltrao/risk_assessment/blob/main/README.md).

The model was trained on a training dataset. The training dataset represents 80% of the total dataset.

First we get the clean training dataset that came from the [03_basic_clean component](https://github.com/vitorbeltrao/risk_assessment/tree/main/components/03_basic_clean); After that, we split the dataset into an array of independent variables (X) and the target vector (y); Then we trained the pipeline using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with cross validation using 5 folds. The best hyperparameters and the best model had already been found in the [experimentation notebook](https://github.com/vitorbeltrao/risk_assessment/blob/main/notebooks/03_workflow_ML.ipynb) and in this model training component 05_train_model, we just instantiate them.

The pipeline used for model training and validation is followed by three steps:

* First, a preprocessing using [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) with only one step for quantitative variables that is [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) class. 

* Finally, in the last step the machine learning model [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

To finalize the whole step, we save the best estimator in a pickle file to go to production.

## Evaluation Data

The data was tested on the remaining 20% ​​of the total dataset, where we never had contact with that dataset. For this, we downloaded the entire inference pipeline stored in the pickle file in the last step and tested it on the test dataset. The evaluation metrics, including sliced data metrics, you can see in the topic below.

## Metrics

**Train and validation metrics:** 

* The mean validation score and mean train score of accuracy in train model step is, respectively: (0.64, 1.0).

**Test metrics:** 

* [metrics_output.txt](https://github.com/vitorbeltrao/risk_assessment/blob/main/components/06_test_model/metrics_output) file.

* [confusion_matrix.png](https://github.com/vitorbeltrao/risk_assessment/blob/main/components/06_test_model/confusion_matrix.png) file.

* [roc_curve.png](https://github.com/vitorbeltrao/risk_assessment/blob/main/components/06_test_model/roc_curve.png) file.

* [slice_output.txt](https://github.com/vitorbeltrao/risk_assessment/blob/main/components/06_test_model/slice_output) file.

## Ethical Considerations

Remembering that the model should be used only for didactic purposes, not for professional purposes.

## Caveats and Recommendations

The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems.