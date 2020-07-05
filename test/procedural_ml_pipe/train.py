'''
train module contains all functions
to train model
'''

#Preprocessing
from .preprocess import *

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

#################
# Training model#
#################

def train():

    logging.basicConfig(filename='train.log', format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)
    #Load and prepare data
    data = data_loader('insurance_claims.csv')
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
    X_train, X_test, y_train, y_test = data_splitter(data, config['target'])
    X_train = selector(X_train, config['features_selected'])
    scaler = scaler_trainer(X_train, './')
    X_train = scaler.transform(X_train)
    #Train the model
    model_trainer(X_train, y_train, './')
    
    logging.info('Training finished!')

if __name__ == '__main__':
    train()