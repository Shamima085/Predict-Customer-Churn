# library doc string
'''
    Author : Mortadha Teffaha
    Date : 2021-04-22
    Imported all required libraries .Implemented all given functions to predict customer churn .
    Refactored the encoder_helper function to make the code efficient.

'''

# import libraries
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


PATH = "./data/bank_data.csv"


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    df = pd.read_csv(pth)
    return df


import_data(PATH)

data_frame = import_data(PATH)
data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
    lambda val: 0 if val == "Existing Customer" else 1)
SAVE_IMG_PATH = "./images/eda/"


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    churn_data = data_frame['Churn']
    plt.figure(figsize=(20, 10))

    churn_data.hist()
    plt.savefig(SAVE_IMG_PATH + 'churn_distribution.png')
    plt.close(fig=None)

    df['Customer_Age'].hist()
    plt.savefig(SAVE_IMG_PATH + 'customer_age_distribution.png')
    plt.close(fig=None)

    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(SAVE_IMG_PATH + 'marital_status_distribution.png')
    plt.close(fig=None)

    sns.displot(df['Total_Trans_Ct'])
    plt.savefig(SAVE_IMG_PATH + 'total_transaction_distribution.png')
    plt.close(fig=None)

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(SAVE_IMG_PATH + 'heatmap.png')
    plt.close(fig=None)


perform_eda(data_frame)
cat_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


y = data_frame['Churn']
X = pd.DataFrame()


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]
            As respons is optional argument I avoided it.

    output:
            df: pandas dataframe with new columns for
    '''

    category_lst_churn = []

    for cat_columns in category_lst:
        cat_list = []
        cat_groups = df.groupby(cat_columns).mean()['Churn']

        for val in df[cat_columns]:
            cat_list.append(cat_groups.loc[val])

        category_lst_churn = cat_columns + "_Churn"
        df[category_lst_churn] = cat_list

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    return X


# print(encoder_helper(data_frame, cat_lst).head())


X = encoder_helper(data_frame, cat_lst)


def perform_feature_engineering(df):
    '''
    input:
            df: pandas dataframe
            response: string of response name
            [optional argument that could be used for naming variables or index y column]
            As respons is optional argument I avoided it.

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_train, y_train)
    pipe.score(x_test, y_test)

    return x_train, x_test, y_train, y_test


perform_feature_engineering(data_frame)


x_axes_train, x_axes_test, y_axes_train, y_axes_test = perform_feature_engineering(
    data_frame)
rfc = RandomForestClassifier(random_state=42)

''' To avoid lbfgs failed to converge (status=1) warning used max-iter=1000 '''

lrc = LogisticRegression(max_iter=1000)


param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
cv_rfc.fit(x_axes_train, y_axes_train)

lrc.fit(x_axes_train, y_axes_train)


y_axes_train_preds_rf = cv_rfc.best_estimator_.predict(x_axes_train)
y_axes_test_preds_rf = cv_rfc.best_estimator_.predict(x_axes_test)

y_axes_train_preds_lr = lrc.predict(x_axes_train)
y_axes_test_preds_lr = lrc.predict(x_axes_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''

    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    plt.rc('figure', figsize=(6, 6))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.05, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.2, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close(fig=None)

    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.05, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.2, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close(fig=None)


classification_report_image(y_axes_train,
                            y_axes_test,
                            y_axes_train_preds_lr,
                            y_axes_train_preds_rf,
                            y_axes_test_preds_lr,
                            y_axes_test_preds_rf)


OUTPUT_PATH = "./images/results/"


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(OUTPUT_PATH + 'feature_importance.png')
    plt.close(fig=None)


# feature_importance_plot(cv_rfc.best_estimator_, X, OUTPUT_PATH)

SAVE_MODEL_PATH = './models/'


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    lrc_plot = plot_roc_curve(lrc, x_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(OUTPUT_PATH + 'roc_curve_result.png')
    plt.close(fig=None)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, SAVE_MODEL_PATH + 'rfc_model.pkl')
    joblib.dump(lrc, SAVE_MODEL_PATH + 'logistic_model.pkl')


train_models(x_axes_train, x_axes_test, y_axes_train, y_axes_test)
