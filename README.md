# Loan Parameter Recommendation System

A machine learning powered recommendation system that predicts loan risk tiers and recommends optimal loan parameters for borrowers based on their financial profile.

---

## Project Overview

This project goes beyond simple loan grade prediction — it actively recommends the best loan term for a given customer profile by simulating multiple scenarios and identifying the option that minimises risk. Built on 100,000 real Lending Club loan records (2007–2018).

**The core question this system answers:**
> *Given a customer's financial profile and desired loan amount, what loan term gives them the best risk outcome?*

---

## Demo Output

```
==================================================
   LOAN PARAMETER RECOMMENDATION SYSTEM
==================================================
Customer Profile:
  FICO Score:     695
  Annual Income:  $55,000
  DTI:            22%
  Loan Amount:    $30,000
  Purpose:        home_improvement
  Home Ownership: RENT

Scenario Analysis:
  Loan Amount     Term         Risk Tier            Confidence
  ------------------------------------------------------------
  $30,000         36 months    Medium Risk          60.3%
  $30,000         60 months    High Risk            87.7%
  $15,000         36 months    Low Risk             54.0% ← RECOMMENDED

RECOMMENDATION:
  Loan Amount: $15,000
  Term:        36 months
  Risk Tier:   Low Risk
==================================================
```

---

## Approach

Three approaches were built and compared:

| Approach | Accuracy | Description |
|---|---|---|
| Rule-Based Baseline | 35% | Manual thresholds using FICO, DTI, loan amount |
| XGBoost Classifier | 68% | ML model with class weights and hyperparameter tuning |
| Recommendation Engine | — | XGBoost + scenario search across loan parameters |

The XGBoost model achieves **68% accuracy** on 3-class risk prediction — nearly double the rule-based baseline — with no data leakage (interest rate excluded as it is assigned post-grade).

---

## Risk Tiers

| Tier | Lending Club Grades | Meaning |
|---|---|---|
| Low Risk | A, B | Strong credit profile |
| Medium Risk | C, D | Moderate credit profile |
| High Risk | E, F, G | Weak credit profile |

---

## Recommendation Logic

```
Step 1 — Try both terms (36 and 60 months) at requested loan amount
Step 2 — If neither gives Low Risk, search lower amounts in $2,500 steps
Step 3 — Recommend: lowest risk tier → shortest term → highest loan amount
Step 4 — If Low Risk never achievable, recommend best available option
```

---

## Tech Stack

- **Python 3.13**
- **XGBoost** — primary classification model
- **Scikit-learn** — preprocessing, train/test split, class weights
- **SHAP** — model explainability
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisation
- **Joblib** — model persistence

---

## Project Structure

```
loan-recsys/
│
├── data/                          # Dataset and processed files (not tracked)
│   ├── accepted_2007_to_2018Q4.csv.gz
│   ├── X_train.csv / X_test.csv
│   ├── y_train.csv / y_test.csv
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
├── notebooks/
│   ├── data_loading.py            # Step 1 — Load and inspect raw data
│   ├── eda.py                     # Step 2 — Exploratory data analysis (8 plots)
│   ├── preprocessing.py           # Step 3 — Feature engineering and cleaning
│   ├── baseline.py                # Step 4 — Rule-based baseline model
│   ├── xgboost_model.py           # Step 5 — XGBoost training and evaluation
│   └── recommender.py             # Step 6 — Recommendation engine
│
├── src/
│   └── plots/                     # Generated EDA and model plots
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) via Kaggle

- 100,000 loan records sampled from 2.2M total records (2007–2018)
- 151 original features reduced to 12 after feature selection

**Key features used:**

| Feature | Description |
|---|---|
| `fico_avg` | Average FICO credit score |
| `loan_amnt` | Requested loan amount |
| `annual_inc` | Annual income |
| `dti` | Debt-to-income ratio |
| `emp_length` | Years of employment |
| `revol_util` | Credit utilisation rate |
| `revol_bal` | Revolving balance |
| `total_acc` | Total credit accounts |
| `home_ownership` | RENT / MORTGAGE / OWN |
| `purpose` | Loan purpose |
| `term` | 36 or 60 months |

**Note:** `int_rate` (interest rate) was deliberately excluded to prevent data leakage — interest rate is assigned after grade determination, not before.

---

## EDA Highlights

- Grade distribution is imbalanced — B and C dominate (~60%), G is <1%
- FICO score shows strong negative correlation with risk tier
- Loan amounts are right-skewed with spikes at round numbers ($10k, $15k, $20k)
- Debt consolidation accounts for ~55% of all loan purposes
- `loan_amnt` and `installment` are highly correlated (0.94) — `installment` dropped
- DTI shows weak standalone relationship with grade but contributes in combination

---

## Model Details

**XGBoost Hyperparameters:**
```python
XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1
)
```

**Class Imbalance Handling:**
```python
sample_weights = compute_sample_weight(
    class_weight={0: 1, 1: 1, 2: 3},  # 3x weight for High Risk
    y=y_train
)
```

**Classification Report:**
```
              precision    recall  f1-score   support
    Low Risk       0.76      0.79      0.77      9653
 Medium Risk       0.65      0.60      0.62      8508
   High Risk       0.43      0.50      0.46      1839
    accuracy                           0.68     20000
```

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to validate that model decisions align with domain knowledge:

- `fico_avg` — high FICO pushes away from Medium/High Risk ✅
- `term_60` — 60 month term increases risk probability ✅
- `loan_amnt` — higher loan amount increases risk ✅
- `dti` — higher debt burden increases risk ✅

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/Shirin0511/loan-recsys.git
cd loan-recsys

# Create virtual environment
python -m venv mlenv
mlenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

---

## Running the Project

Run each step in order:

```bash
python notebooks/data_loading.py      # Verify dataset loads correctly
python notebooks/eda.py               # Generate EDA plots
python notebooks/preprocessing.py    # Clean and prepare data
python notebooks/baseline.py         # Run rule-based baseline
python notebooks/xgboost_model.py    # Train and evaluate XGBoost
python notebooks/recommender.py      # Run recommendation engine
```

---

## Key Learnings

- **Data leakage prevention** — identifying and excluding post-loan features like `int_rate`
- **Class imbalance handling** — using sample weights to improve High Risk recall from 24% to 50%
- **Feature engineering** — deriving `fico_avg` from range columns, rare label encoding for purpose
- **Model explainability** — using SHAP to validate business logic behind predictions
- **Recommendation system design** — converting a classifier into a scenario-based recommender

---

## Author

**Shirin Gupta**
MSc Artificial Intelligence — Queen's University Belfast
[GitHub](https://github.com/Shirin0511) | [LinkedIn](https://linkedin.com/in/shirin-gupta)
