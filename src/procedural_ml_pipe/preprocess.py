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
    :return: DataFrame
    '''
    data.drop[columns_to_drop, axis=1, inplace=True]
    data.rename(columns={"capital-gains": "capital_gains", 
                "capital-loss": "capital_loss"}, inplace=True)
    return data

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

def umbrella_encoder(data, umbrella):
    '''
    Filter for data with positive umbrella limit
    :params data, umbrella
    :return DataFrame
    '''
    flt = data[umbrella]>=0
    return data[flt]

def numerical_encoder(data, var, ):
    pass
    # #discrete variables
    # if var == 'policy_deductable':
    #     bins = list(np.linspace(0,2000, 5, dtype = int))
    #     bin_labels = ['0-500', '501-1000', '1001-1500', '1501-2000']
    #     new_variable_name = "_".join(['policy_deductable', 'groups'])
    #     data[new_variable_name] = pd.cut(data['policy_deductable'], bins = bins, labels = bin_labels)
    #     data.drop('policy_deductable', axis=1, inplace=True)
    
    # bin_labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
    # bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # new_variable_name = "_".join(['age', 'groups'])
    # data[new_variable_name] = pd.cut(data['age'], bins = bins, labels = bin_labels, include_lowest = True)
    # data.drop('age', axis=1, inplace=True)
    
    # bins = list(np.linspace(0,2500, 6, dtype = int))
    # bin_labels = ['very low', 'low', 'medium', 'high', 'very high']
    # new_variable_name = "_".join(['policy_annual_premium', 'groups'])
    # data[new_variable_name] = pd.cut(data['policy_annual_premium'], bins = bins, labels=bin_labels)
    # data.drop('policy_annual_premium', axis=1, inplace=True)
    
    # bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # bin_labels = ['0-50','51-100','101-150','151-200','201-250','251-300','301-350','351-400','401-450','451-500']
    # new_variable_name = "_".join(['months_as_customer', 'groups'])
    # data[new_variable_name] = pd.cut(data['months_as_customer'], bins = 10, labels = bin_labels, include_lowest= True)
    # data.drop(['months_as_customer'], axis=1, inplace=True) 
    
    








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
    
