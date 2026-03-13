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




