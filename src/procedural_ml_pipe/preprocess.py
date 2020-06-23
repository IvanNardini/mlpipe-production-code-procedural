# Data Preparation
import pandas as pd
import numpy as np

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Model Deployment
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Preprocessing functions

def data_loader(datapath):
    '''
    Load the data for training
    :params: datapath
    :return: DataFrame
    '''
    return pd.read_csv(datapath)

def data_preparer(data, columns_to_drop):
    '''
    Drop and Rename columns
    :params: data, columns_to_drop
    :return: data
    '''
    data.drop[columns_to_drop, axis=1, inplace=True]
    data.rename(columns={"capital-gains": "capital_gains", 
                "capital-loss": "capital_loss"}, inplace=True)
    return data

def data_splitter(data, target):
    '''
    Split data in train and test samples
    :params: DataFrame, target name
    :return: X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        data[target],
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def missing_imputer(data, var, replace='missing'):
    '''
    Imputes '?' character with 'missing' label
    :params: data, var, replace
    :return: Series
    '''
    return data[var].replace('?', replace)

def target_encoder(data, target):
    '''
    Encodes target variable
    :params data, target
    :return: Series
    '''
    target_labels = set(data[target])
    target_labels_dic = {label: index for index, label in enumerate(target_labels, 0)}
    return data[target].map(target_labels_dic).astype('category')

def umbrella_encoder(data, var, umbrella)
