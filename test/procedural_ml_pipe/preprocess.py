'''
preprocess.py module contains all functions
train and score steps needs
'''

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
import onnxruntime as rt

#Utils
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Preprocessing functions
def loader(datapath):
    '''
    Load the data for training
    :params: datapath
    :return: DataFrame
    '''
    return pd.read_csv(datapath)

def dropper(data, columns_to_drop):
    '''
    Drop columns
    :params: data, columns_to_drop
    :return: DataFrame
    '''
    data.drop(columns_to_drop, axis=1, inplace=True)
    return data

def renamer(data, columns_to_rename):
    '''
    Rename columns
    :params: data, columns_to_rename
    :return: DataFrame
    '''
    data.rename(columns=columns_to_rename, inplace=True)
    return data

def anomalizier(data, anomaly_var):
    '''
    Drop anomalies 
    :params: data, anomaly_var
    :return: DataFrame
    '''
    flt = data[anomaly_var] >=0
    return data[flt]

def missing_imputer(data, columns_to_impute, replace='missing'):
    '''
    Imputes '?' character with 'missing' label
    :params: data, var, replace
    :return: Series
    '''
    return data[columns_to_impute].replace('?', replace)

def data_splitter(data, target, predictors, test_size, random_state):
    '''
    Split data in train and test samples
    :params: data, target, predictors, test_size, random_state
    :return: X_train, X_test, y_train, y_test
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(data[predictors],
                                                        data[target],
                                                        test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def target_encoder(target, labels_dic):
    '''
    Encode target
    :params: target, labels_dic
    :return: target_encoded
    '''
    target_encoded = target.map(labels_dic).astype('category')
    return target_encoded

def binner(data, var, new_var_name, bins, bins_labels):
    data[new_var_name] = pd.cut(data[var], bins = bins, labels=bins_labels, include_lowest = True)
    data.drop(var, axis=1, inplace=True)
    return data[new_var_name]

# def encoder(data, var, mapping):
#     '''
#     Encode all variables for training
#     :params: data, var, mapping
#     :return: DataFrame
#     '''
#     if var not in data.columns.values.tolist():
#         pass
#     return data[var].map(mapping)

# def dumminizer(data, columns_to_dummies, dummies_meta):
#     '''
#     Generate dummies for nominal variables
#     :params: data, columns_to_dummies, dummies_meta
#     :return: DataFrame
#     '''
#     for var in columns_to_dummies:
#         cat_names = sorted(dummies_meta[var])
#         obs_cat_names = sorted(list(set(data[var].unique())))
#         dummies = pd.get_dummies(data[var], prefix=var)
#         data = pd.concat([data, dummies], axis=1)
#         if obs_cat_names != cat_names: #exception: when label misses 
#             cat_miss_labels = ["_".join([var, cat]) for cat in cat_names if cat not in obs_cat_names] #syntetic dummy
#             for cat in cat_miss_labels:
#                 data[cat] = 0 
#         data = data.drop(var, 1)
#     return data
    
# def scaler_trainer(data, output_path):
#     '''
#     Fit the scaler on predictors
#     :params: data, output_path
#     :return: scaler
#     '''
    
#     scaler = MinMaxScaler()
#     scaler.fit(data)
#     joblib.dump(scaler, output_path)
#     return scaler
  
# def scaler_trasformer(data, scaler):
#     '''
#     Trasform the data 
#     :params: data, scaler
#     :return: DataFrame
#     '''
#     scaler = joblib.load(scaler) 
#     return scaler.transform(data)

# def balancer(data, features_selected, target):
#     '''
#     Balance data with SMOTE
#     :params: data
#     : X, y
#     '''
#     smote = SMOTE(random_state=9)
#     X, y = smote.fit_resample(data[features_selected], data[target])
#     return X,y



# # Training function

# def model_trainer(X_train, y_train, output_path):
#     '''
#     Train the model and store it
#     :params: X_train, y_train, output_path
#     :return: None
#     '''
#     # initialise the model
#     rfor = RandomForestClassifier(max_depth=25, 
#                                   min_samples_split=5, 
#                                   n_estimators=300,
#                                   random_state=8)
       
#     # train the model
#     rfor.fit(X_train, y_train)
    
#     # save the model
#     initial_type = [('features_input', FloatTensorType([1, X_train.shape[1]]))]
#     onnx = convert_sklearn(rfor, name='rf_champion', initial_types=initial_type)
#     with open(output_path, "wb") as f:
#         f.write(onnx.SerializeToString())
#         f.close()
#     return None

# #Scoring function
# def model_scorer(X, model, index):
#     '''
#     Score new data with onnx
#     :params: X, model, index
#     :return: list
#     '''
#     sess = rt.InferenceSession(model)

#     input_name = sess.get_inputs()[0].name
#     label_name = sess.get_outputs()[0].name

#     row_to_score = pd.DataFrame([X.iloc[index]])

#     score = np.array(row_to_score, dtype=np.float32)
#     predictions_onnx = sess.run([label_name], {input_name: score})
#     return predictions_onnx[0]
