'''
train.py module contains all functions
to train model
'''

#Preprocessing
from preprocess import *

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

#################
# Training model#
#################

def train():

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    # Preprocessing
    logging.info('Loading data...')
    data = loader(config['data_ingestion']['data_path'])

    logging.info('Processing data...')
    data = dropper(data, config['preprocessing']['dropped_columns'])
    data = renamer(data, config['preprocessing']['renamed_columns'])
    data = anomalizier(data, 'umbrella_limit')
    data = missing_imputer(data, 
                           config['preprocessing']['missing_predictors'], 
                           replace='missing')

    X_train, X_test, y_train, y_test = data_splitter(data,
                                                     config['data_ingestion']['data_map']['target'],
                                                     config['data_ingestion']['data_map']['predictors'],
                                                     config['preprocessing']['train_test_split_params']['test_size'],
                                                     config['preprocessing']['train_test_split_params']['random_state'])
    # Features Engineering
    logging.info('Engineering features...')

    y_train = target_encoder(y_train, config['features_engineering']['target_encoding'])

    for var, meta in config['features_engineering']['binning_meta'].items():
        binning_meta = meta
        X_train[binning_meta['var_name']] = binner(X_train, var, binning_meta['var_name'], binning_meta['bins'], binning_meta['bins_labels'])

    #Encoding variables
    for var, meta in config['features_engineering']['encoding_meta'].items():
        X_train[var] = encoder(X_train, var, meta)
    
    #Create Dummies
    data = dumminizer(data, config['features_engineering']['nominal_predictors'])

    print(data.columns())

    # #Scaling data
    # logging.info('Scaling Features...')
    # scaler = scaler_trainer(data[config['features']], config['paths']['scaler_path'])
    # data[config['features']] = scaler_trasformer(data[config['features']], config['paths']['scaler_path'])

    # #Balancing sample
    # logging.info('Oversampling with SMOTE...')
    # X, y = balancer(data, config['features_selected'], config['target'])

    # #Split and scale data
    # logging.info('Splitting Data for Training...')
    # X_train, X_test, y_train, y_test = data_splitter(X, y)
    
    # #Train the model
    # logging.info('Training Model...')
    # model_trainer(X_train, y_train, config['paths']['model_path'])

if __name__ == '__main__':

    import logging
    from collections import Counter
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    train()
    logging.info('Training finished!')