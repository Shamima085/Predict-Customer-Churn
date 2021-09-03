
'''
    Author : Mortadha Teffaha
    Date : 2021-04-22
    Testing churn_library.py file.

'''
import os
import os.path
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    Tested file found or not,files rows and column has or not .
    input:
            data_path: a path to the csv
    output:
            None
    '''
    try:
        test_data_frame = cls.import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
        try:
            assert test_data_frame.shape[0] > 0
            assert test_data_frame.shape[1] > 0
            logging.info(
                "Testing import_data: The file appear to has rows and columns")
        except AssertionError:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
    except FileNotFoundError:
        logging.error("Testing import_data: The file wasn't found")
    except (TypeError, AttributeError, ValueError):
        logging.error("Testing import_data: Wrong argument passed")


def test_eda(data_frame):
    '''
    test perform eda function.
    Tested file found or not, file has elements in rows and columns or not . Path exists or not.
    input:
            data_frame: pandas dataframe
            SAVE_IMG_PATH : a path to save eda figures

     output:
            None

    '''
    try:
        cls.perform_eda(data_frame)
        logging.info("Testing perform_eda: SUCCESS")

        try:
            assert data_frame.shape[0] > 0
            assert data_frame.shape[1] > 0
            logging.info(
                "Testing perform_eda: The file has elements in rows and columns")
        except AssertionError:
            logging.error(
                "Testing perform_eda: The file doesn't appear to have elements rows and columns")

        try:
            d_churn = cls.data_frame['Churn'].isnull().sum()
            assert d_churn == 0
            logging.info("Testing perform_eda: No null value ")
        except FileNotFoundError:
            logging.error("Testing perform_eda: null value exist")

        try:
            if os.path.isfile(cls.SAVE_IMG_PATH + 'churn_distribution.png'):
                logging.info("Testing perform_eda: path exists")
            else:
                logging.error("Testing perform_eda: path not exists")
        except FileNotFoundError:
            logging.error(
                "Testing perform_eda: The path  is not found")
    except FileNotFoundError:
        logging.error("Testing perform_eda: The file wasn't found")
    except (TypeError, AttributeError, ValueError, NameError):
        logging.error("Testing perform_eda: Wrong argument passed")


def test_encoder_helper(data_frame, cat_lst):
    '''
    test encoder helper.
    Tested file found or not, file has elements in rows and columns or not. Path exists or not.
    input:
            data_frame: pandas dataframe
            X_data: pandas dataframe of X values
            cat_lst : list of columns that contain categorical features

     output:
            None
    '''
    try:
        cls.encoder_helper(data_frame, cat_lst)
        x_data = cls.encoder_helper(data_frame, cat_lst)
        logging.info("Testing encoder_helper: SUCCESS")

        try:
            assert x_data.shape[0] > 0
            assert x_data.shape[1] > 0
            logging.info(
                "Testing encoder_helper: The file appear to has rows and columns with churn values")
        except AssertionError:
            logging.error(
                "Testing encoder_helper: The file doesn't have rows and columns with churn values")
        try:
            assert len(cat_lst) > 0
            logging.info(
                "Testing encoder_helper: Columns with categorical values are listed")
        except AssertionError:
            logging.error(
                "Testing encoder_helper:  Columns with categorical values are not listed")

    except FileNotFoundError:
        logging.error("Testing encoder_helper: The file wasn't found")
    except (TypeError, AttributeError, ValueError, NameError):
        logging.error("Testing Testing encoder_helper: Wrong argument passed")
    # except (TypeError, AttributeError, ValueError, NameError):
    #     logging.error("Testing Testing encoder_helper: Wrong argument passed")


def test_perform_feature_engineering(data_frame):
    '''
    test perform_feature_engineering.
    Tested file found or not, data splited or not. Path exists or not.
    input:
            data_frame: pandas dataframe
            OUTPUT_PATH: path to store the figure

    output:
            None
    '''
    try:
        cls.perform_feature_engineering(data_frame)
        x_axes_train, x_axes_test, y_axes_train, y_axes_test = cls.perform_feature_engineering(
            data_frame)
        logging.info("Testing perform_feature_engineering: SUCCESS")

        try:
            assert len(x_axes_train) > 0
            assert len(y_axes_train) > 0
            assert len(y_axes_test) > 0
            assert len(x_axes_test) > 0

            logging.info(
                "Testing perform_feature_engineering: Data splited to train and test ")
        except AssertionError:
            logging.error(
                "Testing perform_feature_engineering: No data in train and test set")

        try:
            if os.path.isfile(cls.OUTPUT_PATH + 'importance.png'):
                logging.info(
                    "Testing perform_feature_engineering: path exists")
            else:
                logging.error(
                    "Testing perform_feature_engineering: path not exists")
        except FileNotFoundError:
            logging.error(
                "Testing perform_feature_engineering: The path doesn't found")
    except FileNotFoundError:
        logging.error(
            "Testing perform_feature_engineering: The file wasn't found")
    except (TypeError, AttributeError, ValueError, NameError):
        logging.error(
            "Testing perform_feature_engineering: Wrong argument passed")


def test_train_models(x_axes_train, x_axes_test, y_axes_train, y_axes_test):
    '''
    test train_models.Tested file found or not. Path exists or not.
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              SAVE_MODEL_PATH: path to store the figure
     output:
              None
    '''
    try:
        cls.train_models(x_axes_train, x_axes_test, y_axes_train, y_axes_test)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing train_models: The file wasn't found")
    except (TypeError, AttributeError, ValueError, NameError):
        logging.error("Testing train_models: Wrong argument passed")

    try:
        if os.path.isfile(cls.SAVE_MODEL_PATH + 'logistic_model_try.pkl'):
            logging.info("Testing train_models: path exists")
        else:
            logging.error(
                "Testing train_models: The path SAVE_MODEL_PATH doessn't found")
    except FileNotFoundError:
        logging.error(
            "Testing train_models: The path SAVE_MODEL_PATH doessn't found")


if __name__ == "__main__":
    #  pass
    test_import("./data/bank_data.csv")
    test_import("./n0_data/bank_data.csv")
    test_eda(cls.data_frame)
    test_encoder_helper(cls.data_frame, cls.cat_lst)
    test_perform_feature_engineering(cls.data_frame)
    test_train_models(
        cls.x_axes_test,
        cls.y_axes_test,
        cls.y_axes_train,
        cls.SAVE_MODEL_PATH)
    test_train_models(
        cls.x_axes_train,
        cls.x_axes_test,
        cls.y_axes_train,
        cls.y_axes_test)
