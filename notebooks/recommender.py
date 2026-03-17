import pandas as pd
import numpy as np
import joblib

# Loading Saved Model, Scaler and Feature Columns
model = joblib.load('data/xgboost_model.pkl')
scaler = joblib.load('data/scaler.pkl')
feature_cols = joblib.load('data/feature_col.pkl')

# Risk Mapping

risk_mapping = {0:'Low Risk', 1:'Medium Risk', 2:'High Risk'}

# Numerical Features to Scale 
num_features = ['loan_amnt', 'annual_inc', 'dti',
                'fico_avg', 'revol_bal', 'revol_util',
                'total_acc', 'emp_length']



def build_scenario(customer, loan_amnt, term):

    """
    
    Build a single scenario row matching the exact feature columns
    the model was trained on.
    
    """
    row = {col: 0 for col in feature_cols}

    # Fill numerical feature

    row['loan_amnt'] = loan_amnt
    row['annual_inc'] = customer['annual_inc']
    row['dti'] = customer['dti']
    row['fico_avg'] = customer['fico_avg']
    row['revol_bal'] = customer['revol_bal']
    row['revol_util'] = customer['revol_util']
    row['total_acc'] = customer['total_acc']
    row['emp_length'] = customer['emp_length']

    # Setting term, purpose and ownership binary values

    if term == 60:
        if 'term_60' in row:
            row['term_60'] = 1

    purpose = f"purpose_{customer['purpose']}"
    if purpose in row:
        row[purpose] = 1


    ownership_type = f"home_ownership_{customer['home_ownership_']}"    
    if ownership_type in row:
        row[ownership_type] = 1

    return pd.DataFrame([row])


def scale_scenario(df):

    """Scale numerical features using saved scaler."""

    df_copy = df.copy()
    df_copy[num_features] = scaler.transform(df_copy[num_features])
    return df_copy



def predict_risk(customer, loan_amnt, term):

    scenario = build_scenario(customer, loan_amnt, term)
    scaled_scenario = scaled_scenario(scenario)
    prediction = model.predict(scaled_scenario)[0]
    prob = model.predict_proba(scaled_scenario)[0]
    return prediction, prob


def recommender(customer):

    """
    Given a customer profile:
    1. Try both terms with requested loan amount
    2. If both High Risk — try lower amounts in $2500 steps
    3. Return full scenario table and best recommendation
    """

    requested_amt = customer['loan_amnt']
    terms = [36, 60]
    results = []

    # Step -1 Looping across both term values to find the risk related to each term
    for term in terms:
        risk, proba = predict_risk(customer, requested_amt, term)
        results.append(
            {
                'loan_amnt' : requested_amt,
                'term' : term,
                'risk' : risk,
                'risk_tier' : risk_mapping[risk],
                'probability' : f"{max(proba)*100:.1f}%"
            }
        )


    # Step- 2 if both are in high risk, then reducing loan amt

    both_high_risk = all(r['risk_tier']==2 for r in results)

    if both_high_risk:
        safer_category = False
        updated_amt = requested_amt - 2500

        while updated_amt >= 2500 and not safer_category:
            for term in terms:
                risk, proba = predict_risk(customer, updated_amt, term)
                if risk < 2:
                    results.append(
                    {
                    'loan_amnt' : requested_amt,
                    'term' : term,
                    'risk' : risk,
                    'risk_tier' : risk_mapping[risk],
                    'probability' : f"{max(proba)*100:.1f}%"
                    }
                )
                    safer_category = True
            
            
            if not safer_category:    
                updated_amt = updated_amt - 2500  


    # Step-3 Finding best recommendation
    # Priority: Low Risk > Medium Risk > High Risk
    # Within same tier: prefer shorter term (36 months)
    # Within same tier and term: prefer higher loan amount

    sorted_results= sorted(results, key = lambda x: {
        x['risk_tier'], x['term'], -x['loan_amnt']
    })   


    best = sorted_results[0]
    return results, best





