# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**Explain the problem statement:This dataset contains data on bank customers and if they chose to get a loan. We seek to predict who would be most likely to get a loan based on the charateristics on a individual. This is more of a regression type of problem because it is continuous data.** 

**Explain the solution: The best performing model was a MaxAbs Scaler, Extreme Random with the Azure Auto Software, with a accuracy of .84795.  While the jupyter notebook code made and tested a few models, it gave their best preforming model of StandardScalerWrapper RandomForest and MaxAbsScaler, ExtremeRandomTrees with get an accuracy of .540. Spearman_correlation metric, and a timeout push of 2 minutes was used for this notebook.  The top 3 key features for the notebook based platform to be predicted were first "housing", with next "job_admin" and "month" being nearly equal.  The top 3 key features for the Azure platform were first "cons.conf", with "age" and "duration" being nearly equal.**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

|Property| Value in this example |Description|
|----|----|---|
|**iteration_timeout_minutes**|2|Time limit in minutes for each iteration. Reduce this value to decrease total runtime.|
|**experiment_timeout_minutes**|30|Maximum amount of time in hours that all iterations combined can take before the experiment terminates.|
|**enable_early_stopping**|True|Flag to enable early termination if the score is not improving in the short term.|
|**primary_metric**| spearman_correlation | Metric that you want to optimize. The best-fit model will be chosen based on this metric.|
|**featurization**| auto | By using auto, the experiment can preprocess the input data (handling missing data, converting text to numeric, etc.)|
|**verbosity**| logging.INFO | Controls the level of logging.|
|**n_cross_validations**|5|Number of cross-validation splits to perform when validation data is not specified.|

The pipeline model is ranked off the spearman_correlation, which measures the strength and direction of association between two ranked variables.
**What are the benefits of the parameter sampler you chose?**
I could be specific on my use of hyperparameter tuning.  I used verbosity logging to see how the machine was working.  I used featurization to double check the data and autofix for any errors in the process.  I used learning at a percentage rate to help the machine learn from past data, iterations, or mistakes.  I also used specific batch sizes to help restict underfitting or overfitting of the specific data to the model.
**What are the benefits of the early stopping policy you chose?** 
This early stopping has two main benefits, saving computer power/memory for better tasks, and reducing the possibly of data loss from the Udacity-Azure platform in timing out.  There was a limit of 3 hours max given for each login with Azure.
## AutoML
**Describe the model and hyperparameters generated by AutoML.**
**MaxAbs Scaler, Extreme Random:**
"This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity (scikit-learn.org)."

**MaxAbsScaler, Extreme Random Trees:**
"Extreme Random Trees do not resample observations when building a tree" and "they do not perform bagging (daviddalpiaz.github.io)."
They do not use the efficient split but rather select a random subset of predictors for each split.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**  The Azure software took 49 runs to complete in finding the best model wile the other notebook based model had 79 runs and about 101 iterations. The notebook model did poorer in comparison to the automated Azure software with regards to accuracy.  The accuracy between the two was at a big notable difference of about 40 points.   

The difference could be related to the test size difference in amount, the amount of models tested, and also the time limit restrictions.  The number of n_cross_validations could have also had a difference.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?
Area for improvement:** 
- Add longer time session from 30 minutes, 
- Get rid of 2 minute iteration restriction, 
- Spearman correlation may not be the mest metric to score, so change the metric **

## Proof of Work and of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

I could not download the best preforming model within the time contraints, however, here are some snapshots from the program between the two comparisons (also see snapshot folder):

Azure workspace model:
![Test Image 1](https://github.com/studybug/Azure-Optimizing/blob/main/snapshots/Azure_Hyper_Run.JPG)
Azure workspace model key features:
![Test Image 1](https://github.com/studybug/Azure-Optimizing/blob/main/snapshots/Modelrank.JPG)
Notebook based workspace model:
![Test Image 1](https://github.com/studybug/Azure-Optimizing/blob/main/snapshots/coderun2.JPG)
Notebook based workspace key features:
![Test Image 1](https://github.com/studybug/Azure-Optimizing/blob/main/snapshots/coderun3.JPG)



