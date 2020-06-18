
import pandas as pd
import lightgbmh
import pickle

def score(age, authorities_contacted, auto_make, auto_model, auto_year, bodily_injuries, capital_gains, capital_loss, collision_type, incident_city, incident_date, incident_hour_of_the_day, incident_location, incident_severity, incident_state, incident_type, injury_claim, insured_education_level, insured_hobbies, insured_occupation, insured_relationship, insured_sex, insured_zip, months_as_customer, number_of_vehicles_involved, police_report_available, policy_annual_premium, policy_bind_date, policy_csl, policy_deductable, policy_number, policy_state, property_claim, property_damage, total_claim_amount, umbrella_limit, vehicle_claim, witnesses):
    "Output: EVENT_PROBABILITY, CLASSIFICATION_LABEL"
    
    # Define the misclassification threshold
    prob_threshold = 0.247

    # Initiate model
    try:
        _ModelPreprocessFit
        _ModelFit

    except NameError:

        _pModelPreprocessingFile = open("/models/resources/viya/<Model-UUID>/preprocessor.pkl", "rb")
        _ModelPreprocessingFit = pickle.load(_pModelPreprocessingFile)
        _pModelPreprocessingFile.close()

        _pModelFile = open("/models/resources/viya/<Model-UUID>/model.pkl", "rb")
        _ModelFit = pickle.load(_pModelFile)
        _pModelFile.close()

    # Construct the input array for scoring
    input_array = pd.DataFrame([[age, authorities_contacted, auto_make, auto_model, auto_year, bodily_injuries,
                                     capital_gains, capital_loss, collision_type, incident_city, incident_date,
                                     incident_hour_of_the_day, incident_location, incident_severity, incident_state,
                                     incident_type, injury_claim, insured_education_level, insured_hobbies,
                                     insured_occupation, insured_relationship, insured_sex, insured_zip,
                                     months_as_customer, number_of_vehicles_involved, police_report_available,
                                     policy_annual_premium, policy_bind_date, policy_csl, policy_deductable,
                                     policy_number, policy_state, property_claim, property_damage, total_claim_amount,
                                     umbrella_limit, vehicle_claim, witnesses]],

                                   columns=['age', 'authorities_contacted', 'auto_make', 'auto_model', 'auto_year', 'bodily_injuries',
                                            'capital_gains', 'capital_loss', 'collision_type', 'incident_city', 'incident_date',
                                            'incident_hour_of_the_day', 'incident_location', 'incident_severity', 'incident_state',
                                            'incident_type', 'injury_claim', 'insured_education_level', 'insured_hobbies', 'insured_occupation',
                                            'insured_relationship', 'insured_sex', 'insured_zip', 'months_as_customer',
                                            'number_of_vehicles_involved', 'police_report_available', 'policy_annual_premium', 'policy_bind_date',
                                            'policy_csl', 'policy_deductable', 'policy_number', 'policy_state', 'property_claim',
                                            'property_damage', 'total_claim_amount', 'umbrella_limit', 'vehicle_claim', 'witnesses'])

    # Transform inputs

    _transformed_inputs = _ModelPreprocessingFit.transform(input_array)

    # Calculate the predicted probabilities
    
    _pred_proba = _ModelFit.predict(_transformed_inputs)

#     # Retrieve the event probability
    EVENT_PROBABILITY = float(_pred_proba)

#     # Determine the predicted target category
    if (EVENT_PROBABILITY >= prob_threshold):
        CLASSIFICATION_LABEL = '1'
    else:
        CLASSIFICATION_LABEL = '0'

    return(EVENT_PROBABILITY, CLASSIFICATION_LABEL)
