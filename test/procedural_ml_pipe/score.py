'''
score module contains all functions
to score new data with trained model
'''

#Preprocessing
from preprocess import *

#####################
# Scoring data model#
#####################

def score(data, config):

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
    
    # Features Engineering
    logging.info('Engineering features...')

    ## Create bins
    for var, meta in config['features_engineering']['binning_meta'].items():
        binning_meta = meta
        data[binning_meta['var_name']] = binner(data, var, 
                                                   binning_meta['var_name'], 
                                                   binning_meta['bins'], 
                                                   binning_meta['bins_labels'])
    ## Encode variables
    for var, meta in config['features_engineering']['encoding_meta'].items():
        data[var] = encoder(data, var, meta)
    ## Create Dummies
    data = dumminizer(data, 
                      config['features_engineering']['nominal_predictors'])
    ## Scale variables
    data[config['features_engineering']['features']] = scaler_transformer(
                           data[config['features_engineering']['features']], 
                           scaler)
    
    #Select features
    data = feature_selector(data, 
                               config['features_engineering']['features_selected'])

    #Score data
    logging.info('Scoring...')
    prediction = model_scorer(data, config['model_training']['model_path'], 1) #score only first row (assumption)

    return prediction

if __name__ == '__main__':

    #For testing the score
    import pandas as pd
    from preprocess import *
    import logging

    #Utils
    import ruamel.yaml as yaml
    import warnings
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

     # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    data = loader(config['data_ingestion']['data_path'])

    X_train, X_test, y_train, y_test = data_splitter(data,
                        config['data_ingestion']['data_map']['target'],
                        config['data_ingestion']['data_map']['variables'],
                        config['preprocessing']['train_test_split_params']['test_size'],
                        config['preprocessing']['train_test_split_params']['random_state'])

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    prediction = score(X_test, config)
    logging.info('Scoring finished!')
    logging.info('The prediction label is {}'.format(prediction))