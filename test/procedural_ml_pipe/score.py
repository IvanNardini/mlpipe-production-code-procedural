'''
score module contains all functions
to score new data with trained model
'''

#Preprocessing
from preprocess import *

#####################
# Scoring data model#
#####################

def score(data):

    data = data.copy()

    ## Drop columns
    data = dropper(data, PREPROCESSING['dropped_columns'])
    ## Rename columns 
    data = renamer(data, PREPROCESSING['renamed_columns'])
    ## Remove anomalies
    data = anomalizier(data, 'umbrella_limit')
    ## Impute missing
    data = missing_imputer(data, 
                           PREPROCESSING['missing_predictors'], 
                           replace='missing')
    
    # Features Engineering
    logging.info('Engineering features...')

    ## Create bins
    for var, meta in FEATURES_ENGINEERING['binning_meta'].items():
        binning_meta = meta
        data[binning_meta['var_name']] = binner(data, var, 
                                                   binning_meta['var_name'], 
                                                   binning_meta['bins'], 
                                                   binning_meta['bins_labels'])
    ## Encode variables
    for var, meta in FEATURES_ENGINEERING['encoding_meta'].items():
        data[var] = encoder(data, var, meta)
    ## Create Dummies
    data = dumminizer(data, 
                      FEATURES_ENGINEERING['nominal_predictors'])
    ## Scale variables
    data[FEATURES_ENGINEERING['features']] = scaler_transformer(
                           data[FEATURES_ENGINEERING['features']], 
                           FEATURES_ENGINEERING['scaler_path'])
    #Select features
    data = feature_selector(data, 
                               FEATURES_ENGINEERING['features_selected'])

    #Score data
    logging.info('Scoring...')
    predictions = model_scorer(data, MODEL_TRAINING['model_path']) 

    return data, predictions

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

    DATA_INGESTION = config['data_ingestion']
    PREPROCESSING = config['preprocessing']
    FEATURES_ENGINEERING = config['features_engineering']
    MODEL_TRAINING = config['model_training']

    data = loader(DATA_INGESTION['data_path'])

    X_train, X_test, y_train, y_test = data_splitter(data,
                        DATA_INGESTION['data_map']['target'],
                        DATA_INGESTION['data_map']['variables'],
                        PREPROCESSING['train_test_split_params']['test_size'],
                        PREPROCESSING['train_test_split_params']['random_state'])

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    X_test, predictions = score(X_test)
    logging.info('Scoring finished!')

    y_test = target_encoder(y_test, 
                            FEATURES_ENGINEERING['target_encoding'])

    MODEL = MODEL_TRAINING['model_path']
    print()
    print("*"*20)
    print("Model Predictions".center(20, '*'))
    print("*"*20)
    print()
    print('First 10 prediticions are: {}'.format(predictions[:10]))
    print()   
    print("*"*20)
    print("Model Assessment".center(20, '*'))
    print("*"*20)
    model_evaluator(MODEL, X_test, y_test)
    