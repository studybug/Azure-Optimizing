#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core import Workspace, Experiment

ws = Workspace.get(name="quick-starts-ws-120457")
exp = Experiment(workspace=ws, name="quick-starts-ws-120457")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()


# In[2]:


import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core import Run, Workspace
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
import azureml.dataprep as dprep
from azureml.core.dataset import Dataset

from azureml.core.compute import ComputeTarget, AmlCompute

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

AmlCompute.supported_vmsizes(workspace = ws)

compute_config = AmlCompute.provisioning_configuration(vm_size="Standard_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)


# Choose a name for your cluster.
amlcompute_cluster_name = "gpucluster"

found = False
# Check if this compute target already exists in the workspace.
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'AmlCompute':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]

if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                max_nodes = 4)

    # Create the cluster.\n",
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)

     # For a more detailed view of current AmlCompute status, use get_status().


# In[3]:


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform
from azureml.train.estimator import Estimator
import os

from azureml.train.hyperdrive import normal, uniform, choice

# Specify parameter sampler
ps = RandomParameterSampling( {
        "learning_rate": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    }
)

# Specify a Policy, check job every 2 iterations
policy =  BanditPolicy(evaluation_interval=2, slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")


# workspaceblobstore is the default blob storage
#src.run_config.source_directory_data_store = "workspaceblobstore"

# Create a SKLearn estimator for use with train.py
est = SKLearn("./training",
                   script_params=None,
                   compute_target=compute_target,
                   entry_script='train.py',
                   conda_packages=['scikit-learn'])


# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(estimator=est,
                             hyperparameter_sampling=ps,
                             policy=None,
                             primary_metric_name='validation_acc',
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=4,
                             max_concurrent_runs=4)


# In[4]:


# Submit your hyperdrive run to the experiment and show run details with the widget.

hyperdrive_run = exp.submit(hyperdrive_config)


# In[5]:


from azureml.widgets import RunDetails
RunDetails(hyperdrive_run).show()


# In[6]:


#hyperdrive_run.get_metrics()


# In[7]:


import joblib
# Get your best run and save the model from that run.
best_run = hyperdrive_run.get_best_run_by_primary_metric()
#best_run_metrics = best_run.get_metrics()
#parameter_values = best_run.get_details()['runDefinition']['Arguments']

print('Best Run Id: ', best_run)
#print('\n Accuracy:', best_run_metrics['accuracy'])
#print('\n learning rate:',parameter_values[3])
#print('\n keep probability:',parameter_values[5])
#print('\n batch size:',parameter_values[7])


# In[8]:


from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
ds = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

dataset = Dataset.Tabular.from_delimited_files(path=ds)


# In[9]:


def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)


# In[10]:


# Dict for cleaning data
months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

# Clean and one hot encode data
x_df = dataset.to_pandas_dataframe().dropna()
jobs = pd.get_dummies(x_df.job, prefix="job")
x_df.drop("job", inplace=True, axis=1)
x_df = x_df.join(jobs)
x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
contact = pd.get_dummies(x_df.contact, prefix="contact")
x_df.drop("contact", inplace=True, axis=1)
x_df = x_df.join(contact)
education = pd.get_dummies(x_df.education, prefix="education")
x_df.drop("education", inplace=True, axis=1)
x_df = x_df.join(education)
x_df["month"] = x_df.month.map(months)
x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)


# In[11]:


from sklearn.model_selection import train_test_split

#x_df = dataset.to_pandas_dataframe().dropna()
#x_df = clean_data(dataset)
#x_df = clean_data(x_df)
#y_df = x_df.pop("loan")
x_train, y_train, x_test, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=66)


# In[12]:


x_df.head()


# In[13]:


#from train import clean_data

# Use the clean_data function to clean your data.
#train_data = clean_data(dataset)


# In[14]:


from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
'''automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task=experiment_timeout_minutes=30,
    primary_metric=,
    training_data=train_data,
    label_column_name=label,
    n_cross_validations=)6'''

import logging

automl_settings = {
    "iteration_timeout_minutes": 2,
    "experiment_timeout_minutes": 30,
    "enable_early_stopping": True,
    "primary_metric": 'spearman_correlation',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 5
}


automl_config = AutoMLConfig(task='regression',
                             debug_log='automated_ml_errors.log',
                             training_data=x_train,
                             label_column_name="loan",
                             **automl_settings)


# In[15]:


from azureml.core.experiment import Experiment
experiment = Experiment(ws, "quick-starts-ws-120457")
run = experiment.submit(automl_config, show_output=True)

# Submit your automl run


# In[17]:


# Retrieve and save your best automl model.

from azureml.widgets import RunDetails
RunDetails(run).show()

best_run, fitted_model = run.get_output()
print(best_run)
print(fitted_model)

