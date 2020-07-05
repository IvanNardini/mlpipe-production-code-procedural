'''
score module contains all functions
to score new data with trained model
'''

#Preprocessing
from preprocess import *

#Utils
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

#####################
# Scoring data model#
#####################

def score(data):

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)
    
    #Prepare data
    data = data_preparer(data, config['dropped_columns'])
    #Impute missing
    for var in config['missing_predictors']:
        data[var] = missing_imputer(data, var, replace='missing')
    #Binning variables
    for var, meta in config['binning_meta'].items():
        binning_meta = meta
        data[binning_meta['var_name']] = binner(data, var, binning_meta['var_name'], binning_meta['bins'], binning_meta['bins_labels'])
    #Encoding variables
    for var, meta in config['encoding_meta'].items():
        data[var] = encoder(data, var, meta)
    #Create Dummies
    data = dumminizer(data, config['nominal_predictors'])
    #Split and scale data
    scaler = scaler_trasformer(data, './')
    data = scaler.transform(data)
    #Train the model
    model_scorer(data, './')
    
    logging.info('Training finished!')

if __name__ == '__main__':

    import logging

    # X_train, X_test, y_train, y_test = data_splitter(data, config['target'])
    # X_train = selector(X_train, config['features_selected'])

    # logging.basicConfig(filename='train.log', format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    score()

