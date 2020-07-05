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

    #Load and prepare data
    logging.info('Loading data...')
    data = data_loader('insurance_claims.csv')
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

    #Selecting Features
    logging.info('Selecting Features...')
    model_variables = config['features_selected'] + [config['target']]
    data = selector(data, model_variables)

    #Split and scale data
    logging.info('Splitting Data for Training...')
    X_train, X_test, y_train, y_test = data_splitter(data, config['target'])
    
    #Scaling data
    logging.info('Scaling Features...')
    scaler = scaler_trainer(X_train, './')
    X_train = scaler.transform(X_train)

    #Train the model
    logging.info('Training Model...')
    model_trainer(X_train, y_train, './')

if __name__ == '__main__':

    import logging
    import os
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S:%f %p', level=logging.INFO)
    logging.info('Training process started!')
    train()
    logging.info('Training finished!')