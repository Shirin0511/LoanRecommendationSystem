import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import shap
import joblib


# Loading Processed Data

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


print(X_train.columns.to_list())


# 1. Handling Class Imbalance from Class Weights

# class_counts = y_train.value_counts().sort_index()

# total = len(y_train)

# class_weights = {i: min(total / (len(class_counts) * count),10.0)
#                  for i, count in class_counts.items()} 

# sample_weights = y_train.map(class_weights)

sample_weights= compute_sample_weight(class_weight={0:1, 1:1, 2:3}, y=y_train)


# 2. Building XGBoost

print("-----Training XGBoost-----")

model = XGBClassifier(
    n_estimators = 500,
    max_depth = 8,
    learning_rate = 0.05,
    subsample= 0.8,
    colsample_bytree = 0.8,
    min_child_weight=3,
    gamma=0.1,
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

print(f" XGBoost Accuracy: {accuracy*100:.2f}%")
print(f"Baseline Accuracy: 35.00%")
print(f"Improvement: +{(accuracy*100 - 35.00):.2f}%")

#grade_mapping = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}

risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

print("--------Classification Report---------")

print(classification_report(
    y_test, y_pred,
    target_names= ['Low Risk', 'Medium Risk', 'High Risk'],
    zero_division= 0
))

# 4. Confusion Matrix

plt.figure(figsize=(10,8))
cm= confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'Medium Risk', 'High Risk'],
            yticklabels=['Low Risk', 'Medium Risk', 'High Risk'])

plt.title('Confusion Matrix for Predicted Grade vs Actual Grade')
plt.ylabel('Actual Grade')
plt.xlabel('Predicted Grade')
plt.tight_layout()
plt.savefig('src/plots/confusion_matrix.png')
plt.show()

# 5. Feature Importance

plt.figure(figsize=(10,8))
importance = pd.Series(model.feature_importances_, index = X_train.columns)
importance.sort_values(ascending=True).tail(15).plot(kind="barh", color='steelblue')
plt.title('XGBoost Feature Importance')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('src/plots/importance_plot.png')
plt.show()

print("10 Most Important Features")
print(importance.sort_values(ascending=False).head(10))


# 6. SHAP Explanability

explainer = shap.TreeExplainer(model)
model_shap_values = explainer.shap_values(X_test[:500])

if isinstance(model_shap_values, list):
    shap_to_plot= model_shap_values[1]

else:
    shap_to_plot = model_shap_values[:, :, 1]    


plt.figure()
shap.summary_plot(shap_to_plot, X_test[:500],
                  feature_names= X_train.columns.to_list(),
                  show=False)

plt.title('SHAP Values - Medium Risk Class')
plt.tight_layout()
plt.savefig('src/plots/shap_plot.png')
plt.show()

print("SHAP Plots Saved")


joblib.dump(model, 'data/xgboost_model.pkl')
print("Model Saved")



