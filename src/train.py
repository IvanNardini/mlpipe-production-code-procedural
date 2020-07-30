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

def train(data, config):

    data = data.copy()
    
    # Preprocessing
    logging.info('Processing data...')

    ## Drop columns
    data = dropper(data, config['preprocessing']['dropped_columns'])
    ## Rename columns 
    data = renamer(data, config['preprocessing']['renamed_columns'])
    ## Remove anomalies
    data = anomalizier(data, 'umbrella_limit')
    ## Impute missing
    data = missing_imputer(data, 
                           config['preprocessing']['missing_predictors'], 
                           replace='missing')
    ## Split data
    X_train, X_test, y_train, y_test = data_splitter(data,
                        config['data_ingestion']['data_map']['target'],
                        config['data_ingestion']['data_map']['predictors'],
                        config['preprocessing']['train_test_split_params']['test_size'],
                        config['preprocessing']['train_test_split_params']['random_state'])
    
    # Features Engineering
    logging.info('Engineering features...')

    ## Encode target
    y_train = target_encoder(y_train, 
                             config['features_engineering']['target_encoding'])

    ## Create bins
    for var, meta in config['features_engineering']['binning_meta'].items():
        binning_meta = meta
        X_train[binning_meta['var_name']] = binner(X_train, var, 
                                                   binning_meta['var_name'], 
                                                   binning_meta['bins'], 
                                                   binning_meta['bins_labels'])

    ## Encode variables
    for var, meta in config['features_engineering']['encoding_meta'].items():
        X_train[var] = encoder(X_train, var, meta)

    ## Create Dummies
    X_train = dumminizer(X_train, 
                         config['features_engineering']['nominal_predictors'])
    ## Scale variables
    scaler = scaler_trainer(X_train[config['features_engineering']['features']], 
                           config['features_engineering']['scaler_path'])

    X_train[config['features_engineering']['features']] = scaler.transform(
                           X_train[config['features_engineering']['features']], 
                           )
    
    #Select features
    X_train = feature_selector(X_train, 
                               config['features_engineering']['features_selected'])
    
    #Balancing sample
    X_train, y_train = balancer(X_train, y_train, 
                                config['features_engineering']['random_sample_smote'])

    #Train the model
    logging.info('Training Model...')
    model_trainer(X_train,
                  y_train,
                  config['model_training']['RandomForestClassifier']['max_depth'],
                  config['model_training']['RandomForestClassifier']['min_samples_split'],
                  config['model_training']['RandomForestClassifier']['n_estimators'],
                  config['model_training']['RandomForestClassifier']['random_state'],
                  config['model_training']['model_path'])

if __name__ == '__main__':

    import logging
    from collections import Counter
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    logging.info('Loading data...')
    data = loader(config['data_ingestion']['data_path'])

    logging.info('Training process started!')
    train(data, config)
    logging.info('Training finished!')