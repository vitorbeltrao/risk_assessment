# Dynamic Risk Assessment System - v0.0.1

## Table of Contents

1. [Project Description](#description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Using the API](#api)
5. [Model Scoring and Model Drift](#scoring)
6. [Diagnosing and Fixing Operational Problems](#diagnosing)
6. [Model Reporting and Monitoring](#reporting)
7. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="description"></a>

Currently, companies must, in addition to seeking new customers, keep current customers to avoid losses. The idea of this project is precisely to predict whether a customer will stop using the company's services so that it can take the necessary actions to prevent this. The usage is meant for students, academics or research purpose.

To carry out the project, it was necessary an architecture that met the low budget, in addition to being functional and scalable. Therefore, we use the architecture below:

![Risk assessment architecture](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/risk_assessment%20-%20architecture.jpg?raw=true)
***

## Files Description <a name="files"></a>

In "risk_assessment" repository we have:

* **components**: Inside this folder, we have all the files needed to run the entire model pipeline, from raw data collection to final predictions for never-before-seen data. These are the final files for the production environment. Each component is a block in the model that performs some task and in general generates some output artifact to feed the next steps.

* **infrastructure**: Inside that folder, we have the .py files needed to create the entire cloud structure. For this project, it was necessary to create a data lake with the "raw" and "trusted" layers and all this was done in Python by the Google cloud api.

* **notebooks**: Inside this folder are the experimentation and prototyping notebooks for the entire project. Before creating all this current structure, we tested the project hypothesis with notebooks to verify its viability.

* **tests**: Folder containing unit tests to verify that the API created to infer the machine learning model is returning the correct results.

* **main.py file**: Main script in Python that runs all the components. All this managed by *MLflow* and *Hydra*.

* **ml_api.py file**: Script that creates the necessary methods for creating the API with the *FastAPI* library.

* **conda.yaml file**: File that contains all the libraries and their respective versions so that the system works perfectly.

* **config.yaml**: This is the file where we have the *Hydra* environment variables necessary for the components in main.py to work.

* **environment.yaml**: This file is for creating a virtual *conda* environment. It contains all the necessary libraries and their respective versions to be created in this virtual environment.

* **model_card.md file**: Documentation of the created machine learning model.
***

## Running Files <a name="running"></a>

### Clone the repository

Go to [risk_assessment](https://github.com/vitorbeltrao/risk_assessment) and click on Fork in the upper right corner. This will create a fork in your Github account, i.e., a copy of the repository that is under your control. Now clone the repository locally so you can start working on it:

`git clone https://github.com/[your_github_username]/risk_assessment.git`

and go into the repository:

`cd risk_assessment`

### Create the environment

Make sure to have conda installed and ready, then create a new environment using the *environment.yaml* file provided in the root of the repository and activate it. This file contain list of module needed to run the project:

`conda env create -f environment.yaml`
`conda activate risk_assessment`

### Get API key for Weights and Biases

Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

`wandb login [your API key]`

You should see a message similar to:

`wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc`

### The configuration

The parameters controlling the pipeline are defined in the `config.yaml` file defined in the root of the repository. We will use Hydra to manage this configuration file.

Open this file and get familiar with its content. Remember: this file is only read by the `main.py` script (i.e., the pipeline) and its content is available with the `go` function in `main.py` as the `config` dictionary. For example, the name of the project is contained in the `project_name` key under the `main` section in the configuration file. It can be accessed from the `go` function as `config["main"]["project_name"]`.

### Running the entire pipeline or just a selection of steps

In order to run the pipeline when you are developing, you need to be in the root of the repository, then you can execute this command:

`mlflow run .`

This will run the entire pipeline.

If you want to run a certain steps you can use the examples of command bellow:

`mlflow run . -P steps=upload_raw_data`

This is useful for testing whether steps that have been added or developed can be performed or not.

If you want to run multiple steps (i.e.: `upload_raw_data` and the `upload_trusted_data` steps), you can similarly do:

`mlflow run . -P steps=upload_raw_data,upload_trusted_data`

> NOTE: Make sure the previous artifact step is available in W&B. Otherwise we recommend running each step in order.

> NOTE: To change some environment variable inside the components, you must do it inside the component folder in `MLproject` file. This is where environment variables are instantiated.

### Run existing pipeline

We can directly use the existing pipeline to do the training process without the need to fork the repository. All it takes to do that is to conda environment with MLflow and wandb already installed and configured. To do so, all we have to do is run the following command:

`mlflow run -v [pipeline_version] https://github.com/vitorbeltrao/risk_assessment.git`

`[pipeline_version]` is a release version of the pipeline. For example this repository has currently been released for version `0.0.1`. So we need to input `0.0.1` in place of `[pipeline_version]`.
***

## Using the API <a name="api"></a>
***

## Model Scoring and Model Drift <a name="scoring"></a>

To check the model score and check the model drift, we are doing it by the following process:

![The model scoring process](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/The_model_scoring_proces.png?raw=true)

Model scoring should happen at regular intervals. You should read fresh data, make predictions with a deployed model, and measure the model's errors by comparing predicted and actual values.

If your model begins to perform worse than it had before, then you're a victim of model drift. When model drift occurs, you need to retrain and re-deploy your model.

The file containing the detailed evaluation metrics, including the historical records of the test data evaluation, to verify the model drift, are being detailed in the [model card](https://github.com/vitorbeltrao/risk_assessment/blob/main/model_card.md).



