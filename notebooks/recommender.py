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