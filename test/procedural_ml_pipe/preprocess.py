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
    data.drop(columns_to_drop, axis=1, inplace=True)
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

def binner(data, var, new_var_name, bins, bins_labels):
    data[new_var_name] = pd.cut(data[var], bins = bins, labels=bins_labels)
    data.drop(var, axis=1, inplace=True)
    return data[new_var_name]

def encoder(data, var, mapping):
    '''
    Encode all variables for training
    :params: data, var, mapping
    :return: DataFrame
    '''
    return data[var].map(mapping)

def dumminizer(data, columns_to_dummies):
    '''
    Generate dummies for nominal variables
    :params: data, columns_to_dummies
    :return: DataFrame
    '''
    data = pd.get_dummies(data, columns=columns_to_dummies)
    return data

def selector(data, features_selected):
    '''
    Select Features
    :params: data, features_selected
    :return: DataFrame
    '''
    return data[features_selected]

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

def scaler_trainer(data, output_path):
    '''
    Fit the scaler on predictors
    :params: data, output_path
    :return: scaler
    '''
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, output_path)
    return scaler
  
def scaler_trasformer(data, scaler):
    '''
    Trasform the data 
    :params: data, scaler
    :return: DataFrame
    '''
    scaler = joblib.load(scaler) 
    return scaler.transform(data)
    
def model_trainer(data, target, output_path):
    '''
    Train the model and store it
    :params: data, target, output_path
    :return: None
    '''
    # initialise the model
    rfor = RandomForestClassifier(max_depth=25, 
                                  min_samples_split=5, 
                                  n_estimators=300,
                                  random_state=8)
       
    # train the model
    rfor.fit(data, target)
    
    # save the model
    initial_type = [('features_input', FloatTensorType([1, data.shape[1]]))]
    onnx = convert_sklearn(rfor, name='rf_champion', initial_types=initial_type)
    with open(output_path + "rf_champion.onnx", "wb") as f:
        f.write(onnx.SerializeToString())
        f.close()
    return None

# def model_scorer(data, model):
#     '''
#     Score new data with onnx
#     :params: data, model
#     :return: list
#     '''
#     sess = rt.InferenceSession(model)

#     input_name = sess.get_inputs()[0].name
#     label_name = sess.get_outputs()[0].name

#     score = np.array(data[:1], dtype=np.float32)
#     predictions_onnx = sess.run([label_name], {input_name: score})
#     return predictions_onnx[0]

if __name__ == '__main__':

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

    #Utils
    import joblib
    import ruamel.yaml as yaml
    import warnings
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    data = data_loader('insurance_claims.csv')
    data = data_preparer(data, config['dropped_columns'])

    for var in config['missing_predictors']:
        data[var] = missing_imputer(data, var, replace='missing')

    for var, meta in config['binning_meta'].items():
        binning_meta = meta
        data[binning_meta['var_name']] = binner(data, var, binning_meta['var_name'], binning_meta['bins'], binning_meta['bins_labels'])
    
    data = dumminizer(data, config['nominal_predictors'])

    data = selector(data, config['features_selected'])

    X_train, X_test, y_train, y_test = data_splitter(data, config['target'])
    
    scaler = scaler_trainer(X_train, '.')

    X_train = scaler.transform(X_train)

    model_trainer(X_train, y_train, '.')

    print('Finished training')







    
