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
    
    #Prepare data
    logging.info('Preparing data...')
    data = data_preparer(data, config['dropped_columns'])

    #Impute missing
    logging.info('Imputing Missings...')
    for var in config['missing_predictors']:
        data[var] = missing_imputer(data, var, replace='missing')
    
    #Binning variables
    logging.info('Binning Variables...')
    for var, meta in config['binning_meta'].items():
        binning_meta = meta
        data[binning_meta['var_name']] = binner(data, var, binning_meta['var_name'], binning_meta['bins'], binning_meta['bins_labels'])
    
    #Encoding variables
    logging.info('Encoding Variables...')
    for var, meta in config['encoding_meta'].items():
        data[var] = encoder(data, var, meta)
    
    #Create Dummies
    logging.info('Generating Dummies...')
    data = dumminizer(data, config['nominal_predictors'])

    #Select Features
    model_variables = config['features_selected'] + [config['target']]
    data = selector(data, model_variables)

    #Scale data
    logging.info('Scaling Features...')
    data = scaler_trasformer(data, config['paths']['scaler_path'])

    #Score data
    model_scorer(data, config['paths']['model_path'])

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

    data = data_loader(config['paths']['data_path'])

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        data[config['target']])
    
    row_to_score = X_test.loc[1,:]

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    prediction = score(row_to_score)
    logging.info('Scoring finished!')
    logging.info('The prediction label is {}'.format(prediction))