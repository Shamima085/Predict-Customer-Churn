
# **Project Overview**

In this project, I implemented my learnings to identify credit card customers that are most likely to churn. The completed project include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). To test PEP8 guidelines maintened or not installed  the linter and auto-formatter:
`pylint`
`autopep8`
I have done this project in my local machine using visual studio code.

## `churn_library.py`

### **Required Librarirs for `churn_library.py`**
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


Refactored the code of `churn_notebook.ipynb file` to code in churn_library.py. Code of `churn_library.py` completes the process for solving the data science process.

In import data function imported data through the PATH and retured the value . 

In eda function performed eda on data_frame and plotted figure for ['Churn'], ['Customer_Age'], ['Marital_Status'],['Total_Trans_Ct'] columns also the heatmap. Saved these images in `SAVE_IMG_PATH = "./images/eda/"`

In encoder_helper turned each categorical column into a new column with propotion of churn for each category. As an input used data_frame and category_lst(list of columns that contain categorical features). As response was optional I haven't used it.
X = pd.DataFrame() defined empty dataframe further store the new columns with existing columns in X
To make the code efficient listed the columns using for loop. 

This returens data_frame with new columns for.

In perform_feature_engineering function used data_frame X and splited the data in train_test. Performed StandardScaler in train and test data.
This function returns training and testing data of x and y.

To perform classification_report_image in outer scope defined model names and fited train and test data. To avoid lbfgs failed to converge (status=1) warning used max-iter=1000 . In this function done training predictions from logistic regression and random forest also test predictions from logistic regression and random forest .
As an input took (y_train,y_test,y_train_preds_lr,y_train_preds_rf,y_test_preds_lr,y_test_preds_rf)
Saved the report of logistic regression and random forest in "./images/results/" path.

In feature_importance_plot function created and stored the feature importances in OUTPUT_PATH

AS an input used model object containing feature_importances_ which is cv_rfc.best_estimator_
also used pandas dataframe of X values and the defined OUTPUT_PATH.

In train_models function stored model results and saved the best model in SAVE_MODEL_PATH = './models/'. 
plotted a figure of cv_rfc.best_estimator_ referring the both LogisticRegression and random forest saved the figure in "./images/results/"
As an input used training and testing data.

**`churn_library.py` code has been rated at 9.21/10**

## churn_script_logging_and_tests.py 

### **Required Librarirs for `churn_library.py`**
import os
import os.path
import logging
import churn_library as cls

After completing the churn_library.py imported it as cls in churn_script_logging_and_tests.py file.

`churn_script_logging_and_tests.py` is completed with tests for the each input function.
Each function in `churn_script_logging_and_tests.py` is completed with logging for if the function successfully passes the tests or errors.

All log information is stored in a `churn_library.log` file, used easily understandable be and traceable log messages.Also used timestamp so that it helps to understand the log info and error.

To test 4 functions are given:

To test_import function given the both real and wrong data path to check info and error.

To test eda function first checked the file exists or not. Used assertion to check data_frame elements existance using shape. Checked data_frame[churn] null values. Finalyy tried to test the saved page path with wrong png file does it exists or not. 

To test encoder_helper function first checked the file exists or not.Used assestion to check x_data elements existance using shape.Also checked length of cat_lst to know columns with categorical values are listed or not. To handle (TypeError, AttributeError, ValueError) used an except.

To test perform_feature_engineering function first checked the file exists or not.Checked the length of all splited data using assertion.Finaly tried to test the saved page path with wrong png file does it exists or not. To handle (TypeError, AttributeError, ValueError) used an except.

To test train_models function first checked the file exists or not.tried to test the saved page path with wrong png file does it exists or not.To handle (TypeError, AttributeError, ValueError) used an except.

**`churn_script_logging_and_tests.py` code has been rated at 10.00/10**

Author : Shamima Sultana
Date : 2021-04-22





