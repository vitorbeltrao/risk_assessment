# Dynamic Risk Assessment System - v1.0.0

## Table of Contents

1. [Project Description](#description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Using the API](#api)
5. [Model and Data Diagnostics](#diagnostics)
6. [Orchestration](#orchestration)
7. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="description"></a>

Currently, companies must, in addition to seeking new customers, keep current customers to avoid losses. The idea of this project is precisely to predict whether a customer will stop using the company's services so that it can take the necessary actions to prevent this. The usage is meant for students, academics or research purpose.

To carry out the project, it was necessary an architecture that met the low budget, in addition to being functional and scalable. Therefore, we use the architecture below:

![Risk assessment architecture](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/risk_assessment%20-%20architecture.jpg?raw=true)
***

## Files Description <a name="files"></a>

In "risk_assessment" repository we have:

* **components**: Inside this folder, we have all the files needed to run the entire model pipeline, from raw data collection to final predictions for never-before-seen data. These are the final files for the production environment. Each component is a block in the model that performs some task and in general generates some output artifact to feed the next steps. All these components are managed by *MLflow Projects*.

* **infrastructure**: Inside that folder, we have the .py files needed to create the entire cloud structure. For this project, it was necessary to create a data lake with the "raw" and "trusted" layers and all this was done in Python by the Google cloud api. Also, we made a script that moves data from the data lake to a data warehouse. This script is not being used in practice, but it might come in handy at some point.

* **notebooks**: Inside this folder are the experimentation and prototyping notebooks for the entire project. Before creating all this current structure, we tested the project hypothesis with notebooks to verify its viability.

* **tests**: Folder containing unit tests to verify that the API created to infer the machine learning model is returning the correct results. This is tested along the CI/CD mats with *Github Actions* so we only integrate and deploy the model to something that is working.

* **model_data_diagnostics**: Folder that contains two scripts that monitor the model in production, checking the model drift and the data drift every day.

* **prod_deployment_path**: Folder that contains the copy of the last files to be deployed in production or that contains some information relevant to the last model.

* **main.py file**: Main script in Python that runs all the components. All this managed by *MLflow Projects*.

* **ml_api.py file**: Script that creates the necessary methods for creating the API with the *FastAPI* library.

* **scheduler.py**: This is the file that uses the *apscheduler* library to orchestrate our system, more details you can see in the "Orchestration" topic.

* **conda.yaml file**: File that contains all the libraries and their respective versions so that the system works perfectly.

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

To check the application documentation follow the [link](https://risk-assessment-system.onrender.com/docs)

The url to the endpoint for making API requests is: https://risk-assessment-system.onrender.com/risk_assessment_prediction
***

## Model and Data Diagnostics <a name="diagnostics"></a>

### Model Drift

To check the model score and check the model drift, we are doing it by the following process:

![The model scoring process](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/The_model_scoring_proces.png?raw=true)

Model scoring should happen at regular intervals. You should read fresh data, make predictions with a deployed model, and measure the model's errors by comparing predicted and actual values.

If your model begins to perform worse than it had before, then you're a victim of model drift. When model drift occurs, you need to retrain and re-deploy your model.

The file containing the detailed evaluation metrics, including the historical records of the test data evaluation, to verify the model drift, are being detailed in the [model card](https://github.com/vitorbeltrao/risk_assessment/blob/main/model_card.md).

In the *model_data_diagnostics* folder, we create the scripts that check the model drift through three functions: *Raw Comparison Test*, *Parametric Significance Test* and *Non-Parametric Outlier Test*. For more information on what each of these tests does, visit the respective folder with the scripts and see the documentation for the functions.

Finally, after testing these three functions that verify the model drift, we choose by voting whether the model suffered model drift or not, that is, if two of these functions show model drift, then we have model drift and vice versa. If we don't have model drift, then we keep the current model in production; if we have model drift then we must retrain and re-deploy the model. **Our system automatically sends an email to the person in charge, in case the model drift occurs!**

### Data Drift

Data drift also happens at regular intervals. We have a reference dataset, which was the first dataset that we trained, validated and tested the model before going into production and everything went well on that dataset. Over time, more data enters the data lake and the idea here is to compare the entire historical dataset (reference + new data coming in regularly) with the reference dataset that was the first one we trained.

To make this comparison, we used an open source library that is [Evidently](https://www.evidentlyai.com/). For more information read the documentation at the highlighted link. Finally, we generate HTML files with the entire report on the data drift for the user, which you can find in the [folder](https://github.com/vitorbeltrao/risk_assessment/tree/main/model_data_diagnostics) where we are checking the diagnostics.
***

## Orchestration <a name="orchestration"></a>

The orchestration, for our system to work automatically, is all done with the help of the [apscheduler](https://apscheduler.readthedocs.io/en/3.x/) library. For more information read the documentation at the highlighted link. In addition to the apscheduler, we also use the windows task scheduler, for the system to run regularly (in our case daily).

The algorithm that is in the *scheduler.py* file was designed with the following proposal:

* Every day we run the function *upload_raw_data* and *upload_trusted_data* that are in the components folder.

* After running the two previous functions in sequence, we run the two diagnostic files in parallel, referring to the model drift and the data drift.

* If model drift occurs, the model is automatically retrained with the entire available training dataset. If the model drift does not happen, then the pipeline is interrupted and will run again the next day.

**With that, we have a model working almost 100% automatically without much manual intervention on the part of those responsible. Of course, it is recommended that sometimes those responsible take a look at the reports, to verify that everything is fine**.
***

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/v%C3%ADtor-beltr%C3%A3o-56a912178/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://pandascouple.medium.com)

Licensing: [MIT LICENSE](https://github.com/vitorbeltrao/risk_assessment/blob/main/LICENSE)