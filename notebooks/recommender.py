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



def build_scenario(customer, loan_amt, term):

    """
    
    Build a single scenario row matching the exact feature columns
    the model was trained on.
    
    """
    row = {col: 0 for col in feature_cols}

    # Fill numerical feature

    row['loan_amt'] = loan_amt
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






