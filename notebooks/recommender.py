import pandas as pd
import numpy as np
import joblib

# Loading Saved Model, Scaler and Feature Columns
model = joblib.load('data/xgboost_model.pkl')
scaler = joblib.load('data/scaler.pkl')
feature_cols = joblib.load('data/feature_cols.pkl')

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


    ownership_type = f"home_ownership_{customer['home_ownership']}"    
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
    scaled_scenario = scale_scenario(scenario)
    prediction = model.predict(scaled_scenario)[0]
    prob = model.predict_proba(scaled_scenario)[0]
    return prediction, prob


def recommender(customer):

    """
    Given a customer profile:
    1. Try both terms with requested loan amount
    2. If neither scenario is Low Risk, trigger lower amount search
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
                'risk_category' : risk_mapping[risk],
                'confidence' : f"{max(proba)*100:.1f}%"
            }
        )


    # Step- 2 Trigger lower amount search if neither scenario is Low Risk

    neither_low_risk = all(r['risk']>0 for r in results)

    if neither_low_risk:
        found_low_risk = False
        best_alt= None
        updated_amt = requested_amt - 2500

        while updated_amt >= 2500:
            for term in terms:
                risk, proba = predict_risk(customer, updated_amt, term)
                if best_alt is None or risk < best_alt['risk']:
                    best_alt= {
                    'loan_amnt' : updated_amt,
                    'term' : term,
                    'risk' : risk,
                    'risk_category' : risk_mapping[risk],
                    'confidence' : f"{max(proba)*100:.1f}%"
                    }
                            
            
                if risk == 0: # Found low risk
                    results.append(best_alt)
                    found_low_risk = True
                    break

            if found_low_risk:
                break

            updated_amt = updated_amt - 2500

        # if low risk is never found, append the best alt found 
        if not found_low_risk and best_alt is not None:
            results.append(best_alt)    


    # Step-3 Finding best recommendation
    # Priority: Low Risk > Medium Risk > High Risk
    # Within same tier: prefer shorter term (36 months)
    # Within same tier and term: prefer higher loan amount

    sorted_results= sorted(results, key = lambda x: {
        x['risk'], x['term'], -x['loan_amnt']
    })   


    best = sorted_results[0]
    return results, best



def display_recommendation(customer, results, best):

    print("\n" + "="*50)
    print("   LOAN PARAMETER RECOMMENDATION SYSTEM")
    print("="*50)

    print(f"\nCustomer Profile:")
    print(f"  FICO Score:     {customer['fico_avg']}")
    print(f"  Annual Income:  ${customer['annual_inc']:,.0f}")
    print(f"  DTI:            {customer['dti']}%")
    print(f"  Loan Amount:    ${customer['loan_amnt']:,.0f}")
    print(f"  Purpose:        {customer['purpose']}")
    print(f"  Home Ownership: {customer['home_ownership']}")

    print(f"\nScenario Analysis:")
    print(f"  {'Loan Amount':<15} {'Term':<12} {'Risk Tier':<20} {'Confidence'}")
    print(f"  {'-'*60}")
    best_idx = next(i for i, r in enumerate(results)
                    if r['loan_amnt'] == best['loan_amnt']
                    and r['term'] == best['term']
                    and r['risk'] == best['risk'])

    for idx, r in enumerate(results):
        marker = " ← RECOMMENDED" if idx == best_idx else ""
        print(f"  ${r['loan_amnt']:<14,.0f} {r['term']} months{'':<4} "
              f"{r['risk_category']:<20} {r['confidence']}{marker}")

    print(f"\n{'='*50}")
    print(f"RECOMMENDATION:")
    print(f"  Loan Amount: ${best['loan_amnt']:,.0f}")
    print(f"  Term:        {best['term']} months")
    print(f"  Risk Tier:   {best['risk_category']}")
    print(f"  Confidence:  {best['confidence']}")
    print(f"{'='*50}\n")

if __name__ == '__main__': 
    # Customer 1 — Strong profile
    customer1 = {
        'fico_avg': 740,
        'annual_inc': 80000,
        'dti': 12,
        'loan_amnt': 15000,
        'revol_bal': 5000,
        'revol_util': 20,
        'total_acc': 15,
        'emp_length': 8,
        'purpose': 'debt_consolidation',
        'home_ownership': 'MORTGAGE'
    }

    # Customer 2 — Weak profile
    customer2 = {
        'fico_avg': 660,
        'annual_inc': 40000,
        'dti': 28,
        'loan_amnt': 25000,
        'revol_bal': 12000,
        'revol_util': 75,
        'total_acc': 8,
        'emp_length': 2,
        'purpose': 'debt_consolidation',
        'home_ownership': 'RENT'
    }

    # Customer 3 — Medium profile asking for high amount
    customer3 = {
        'fico_avg': 695,
        'annual_inc': 55000,
        'dti': 22,
        'loan_amnt': 30000,
        'revol_bal': 8000,
        'revol_util': 45,
        'total_acc': 11,
        'emp_length': 5,
        'purpose': 'home_improvement',
        'home_ownership': 'RENT'
    }

    for i, customer in enumerate([customer1, customer2, customer3], 1):
        print(f"\n{'#'*50}")
        print(f"# CUSTOMER {i}")
        print(f"{'#'*50}")
        results, best = recommender(customer)
        display_recommendation(customer, results, best)   

