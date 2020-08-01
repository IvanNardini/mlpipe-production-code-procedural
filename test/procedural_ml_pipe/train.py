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

def train(data):

    data = data.copy()

    # Preprocessing
    logging.info('Processing data...')
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

    ## Split data
    X_train, X_test, y_train, y_test = data_splitter(data,
                        DATA_INGESTION['data_map']['target'],
                        PREPROCESSING['predictors'],
                        PREPROCESSING['train_test_split_params']['test_size'],
                        PREPROCESSING['train_test_split_params']['random_state'])
    
    # Features Engineering
    logging.info('Engineering features...')

    ## Encode target
    y_train = target_encoder(y_train, 
                             FEATURES_ENGINEERING['target_encoding'])

    ## Create bins
    for var, meta in FEATURES_ENGINEERING['binning_meta'].items():
        binning_meta = meta
        X_train[binning_meta['var_name']] = binner(X_train, var, 
                                                   binning_meta['var_name'], 
                                                   binning_meta['bins'], 
                                                   binning_meta['bins_labels'])

    ## Encode variables
    for var, meta in FEATURES_ENGINEERING['encoding_meta'].items():
        X_train[var] = encoder(X_train, var, meta)

    ## Create Dummies
    X_train = dumminizer(X_train, 
                         FEATURES_ENGINEERING['nominal_predictors'])
    ## Scale variables
    scaler = scaler_trainer(X_train[FEATURES_ENGINEERING['features']], 
                           FEATURES_ENGINEERING['scaler_path'])

    X_train[FEATURES_ENGINEERING['features']] = scaler.transform(
                           X_train[FEATURES_ENGINEERING['features']], 
                           )
    
    #Select features
    X_train = feature_selector(X_train, 
                               FEATURES_ENGINEERING['features_selected'])
    
    #Balancing sample
    X_train, y_train = balancer(X_train, y_train, 
                                FEATURES_ENGINEERING['random_sample_smote'])

    #Train the model
    logging.info('Training Model...')
    model_trainer(X_train,
                  y_train,
                  MODEL_TRAINING['RandomForestClassifier']['max_depth'],
                  MODEL_TRAINING['RandomForestClassifier']['min_samples_split'],
                  MODEL_TRAINING['RandomForestClassifier']['n_estimators'],
                  MODEL_TRAINING['RandomForestClassifier']['random_state'],
                  MODEL_TRAINING['model_path'])

if __name__ == '__main__':

    import logging
    from collections import Counter
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    DATA_INGESTION = config['data_ingestion']
    PREPROCESSING = config['preprocessing']
    FEATURES_ENGINEERING = config['features_engineering']
    MODEL_TRAINING = config['model_training']

    logging.info('Loading data...')
    data = loader(DATA_INGESTION['data_path'])
    
    logging.info('Training process started!')
    train(data)
    logging.info('Training finished!')