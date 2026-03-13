import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap


# Loading Processed Data

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# 1. Handling Class Imbalance from Class Weights

class_counts = y_train.value_counts().sort_index()

total = len(y_train)

class_weights = {i: total / (len(class_counts) * count)
                 for i, count in class_counts.items()} 

sample_weights = y_train.map(class_weights)


# 2. Building XGBoost

print("-----Training XGBoost-----")

model = XGBClassifier(
    n_estimators = 300,
    max_depth = 6,
    learning_rate = 0.1,
    subsample= 0.8,
    colsample_bytree = 0.8,
    use_label_encoder = False,
    eval_metric = 'mlogloss',
    random_state = 42,
    n_jobs= -1
)

model.fit(
    X_train, y_train,
    sample_weight = sample_weights,
    eval_set=[(X_test, y_test)],
    verbose = 50
)

print("Model Trained")

# 3. Evaluating Model 

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

