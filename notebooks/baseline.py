import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

#Loading the train and test files
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

print(f"Data Loaded: {X_train.shape}")

grade_mapping = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}

# Rule-Based Logic 
# These rules are based on EDA insights:
# - Higher FICO = lower grade (better)
# - Higher int_rate = higher grade (worse)
# - Higher DTI = higher grade (worse)

def rule_based_predictions(row):

    fico = row['fico_avg']
    dti = row['dti']
    loan_amt= row['loan_amnt']

    # Combine signals into a risk score
    # int_rate is the strongest signal from EDA
    # fico_avg is negative (higher fico = lower risk)
    # loan amount is positive ( higher loan amt is risky)

    risk_score =  - (0.9 * fico) + (0.2 * dti) + (0.1 * loan_amt)

    # Mapping risk score to grades
    if risk_score < -0.85:
        return 0
    elif risk_score < -0.05:
        return 1
    elif risk_score < 0.9:
        return 2
    elif risk_score < 1.25:
        return 3
    elif risk_score < 1.6:
        return 4
    elif risk_score < 2.2:
        return 5
    else:
        return 6
    

# Generating predictions    

y_pred = X_test.apply(rule_based_predictions, axis=1)

# Evaluating the Rule based predictor

accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


print("=============Classification Report=============")

print(classification_report(y_test, y_pred,
                            target_names=['A','B','C','D','E','F','G'],
                            zero_division=0))

# Grade Distribution Comparison

print("=======Actual Grade Distribution=======")

actual_dist = pd.Series(y_test).map(grade_mapping).value_counts().sort_index()

print(actual_dist)

print("=======Predicted Grade Distribution======")

pred_dist = pd.Series(y_pred).map(grade_mapping).value_counts().sort_index()

print(pred_dist)


#Saving Baseline Predictions

baseline_results= pd.DataFrame(
    {
        'actual' : y_test.values,
        'predicted' : y_pred.values
    }
)

baseline_results.to_csv('data/baseline_results.csv', index=False)

print("Baseline Predictions Saved")
