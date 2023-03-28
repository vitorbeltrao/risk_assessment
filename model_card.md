# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The Dynamic Risk Assessment System project is based on a machine learning model that predicts whether a customer will stop using the company's services. In this case, we created a model that will make this prediction, so that the company can take some action to prevent the customer from leaving the company, and thus avoid losses.

Prediction task is to determine whether a person makes over 50K a year. We use a RandomForestClassifier using the optimized hyperparameters in scikit-learn 1.2.1. Hyperparameters tuning was realized using GridSearchCV. Optimal parameters used are:

* n_estimators: 50
* max_depth: 10 

Model is saved as a pickle file stored in the wandb. To use the latest model available in production, you must download it via wandb, like this: `model_local_path = run.use_artifact('[your_wandb_username]/[your_wandb_projectname]/[your_wandb_filename]:prod', type='pickle').download()`

Model training and validation was done in component [06_train_model](https://github.com/vitorbeltrao/census-income-forecast/tree/main/components/06_train_model)

Model testing was done in component [07_test_model](https://github.com/vitorbeltrao/census-income-forecast/tree/main/components/07_test_model)

## Intended Use

This model can be used to predict the salary level of an individual based off a handful of attributes. The usage is meant for students, academics or research purpose.

## Training Data

The dataset used was the census, from the UCI Machine Learning Repository. For specific information about the dataset, see the [link](https://archive.ics.uci.edu/ml/datasets/census+income).

The model was trained on a training dataset. The training dataset represents 80% of the total dataset.

First we get the clean training dataset that came from the [04_basic_clean component](https://github.com/vitorbeltrao/census-income-forecast/tree/main/components/04_basic_clean); After that, we split the dataset into an array of independent variables (X) and the target vector (y); Then we trained the pipeline using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with cross validation using 5 folds. The best hyperparameters and the best model had already been found in the experimentation notebook and in this model training component 06_train_model, we just instantiate them.

The pipeline used for model training and validation is followed by three steps:

* First, a preprocessing using [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) with a step for nominal qualitative variables and another step for quantitative variables.

* Then, one more preprocessing step, this time scaling the entire training set using the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) class.

* Finally, in the last step the machine learning model [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

To finalize the whole step, we save the best estimator in a pickle file to go to production.

## Evaluation Data

The data was tested on the remaining 20% ​​of the total dataset, where we never had contact with that dataset. For this, we downloaded the entire inference pipeline stored in the pickle file in the last step and tested it on the test dataset. The evaluation metrics, including sliced ​​data metrics, you can see in this [slice_output.txt](https://github.com/vitorbeltrao/census-income-forecast/tree/main/components/07_test_model) file.

## Metrics

See: 

* The mean validation score and mean train score of f1 in train model step is, respectively: (0.6665742296070268, 0.999980061808394)

* [slice_output.txt](https://github.com/vitorbeltrao/census-income-forecast/tree/main/components/07_test_model) file.

## Ethical Considerations

The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories. Remembering that the model should be used only for didactic purposes, not for professional purposes.

## Caveats and Recommendations

Extraction was done from the 1994 Census database. The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems.