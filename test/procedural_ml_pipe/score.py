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

    #Split and scale data
    logging.info('Scaling Features...')
    scaler = scaler_trasformer(data, './')
    data = scaler.transform(data)

    #Score data
    model_scorer(data, './*.onnx')

if __name__ == '__main__':

    #For testing the score

    from preprocess import *
    import logging

    data = data_loader('insurance_claims.csv')

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    prediction = score(data[:1])
    logging.info('Scoring finished!')
    logging.info('The prediction label is {}'.format(prediction))