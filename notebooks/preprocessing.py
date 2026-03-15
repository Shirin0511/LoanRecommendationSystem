import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Data Loading

df = pd.read_csv(
    'data/accepted_2007_to_2018Q4.csv.gz',
    compression='gzip',
    low_memory=False,
    nrows=100000
)
print(f" Data loaded: {df.shape}")

# 1. Selecting relevant features
features=[
    'loan_amnt', 'term',  'annual_inc', 'purpose',
    'dti', 'fico_range_low', 'fico_range_high', 'open_acc',
    'revol_bal', 'revol_util', 'total_acc', 'home_ownership',
    'emp_length', 'grade'
]


df_new = df[features].copy()

print(f"Shape of new Dataframe: {df_new.shape}")
print(f"Missing values: {df_new.isnull().sum()}")

# 2. Dropping redundant columns 

df_new['fico_avg'] = (df_new['fico_range_low'] + df_new['fico_range_high'])/2

df_new.drop(['fico_range_high','fico_range_low','open_acc'],axis=1,inplace=True)

print(f"Shape after dropping columns: {df_new.shape}")

# 3. Handling Missing Values
missing_pct = (((df_new.isnull().sum())/len(df_new))*100).round(2)
print(f"Missing Percentages: ", missing_pct[missing_pct>0])

# Filling numerical columns with median
num_cols = ['loan_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'fico_avg']

for col in num_cols:
    df_new[col] = df_new[col].fillna(df_new[col].median())


#Filling actegorical columns with mode
cat_cols = ['purpose', 'home_ownership', 'emp_length']

for col in cat_cols:
    df_new[col] = df_new[col].fillna(df_new[col].mode()[0])


print(df_new.isnull().sum())    


# 4. Clean specific columns

# Changing int_rate to float and removing the % sign
#df_new['int_rate']= df_new['int_rate'].astype(str).str.replace('%','').astype(float)  # commented as this is turing into data leakage

# Changing the term to remove the months keyword from it and keeping only numeric value
df_new['term'] = df_new['term'].astype(str).str.extract(r'(\d+)').astype(int)
df_new['term'] = pd.to_numeric(df_new['term'], errors='coerce')
df_new['term'] = df_new['term'].fillna(df_new['term'].mode()[0])
df_new['term'] = df_new['term'].astype(int)

# Extracting only the numeric part from emp_length
df_new['emp_length'] = df_new['emp_length'].astype(str).str.extract(r'(\d+)')
df_new['emp_length'] = pd.to_numeric(df_new['emp_length'], errors='coerce')
df_new['emp_length'].fillna(df_new['emp_length'].median(), inplace=True)


# 5. Handling Outliers

dti_cap = df_new['dti'].quantile(0.99)
df_new['dti'] = df_new['dti'].clip(upper=dti_cap)

inc_cap = df_new['annual_inc'].quantile(0.99)
df_new['annual_inc'] = df_new['annual_inc'].clip(upper=inc_cap)

revol_cap = df_new['revol_bal'].quantile(0.99)
df_new['revol_bal'] = df_new['revol_bal'].clip(upper=revol_cap)

print(f"DTI capped at: {dti_cap:.2f}")
print(f"Income capped at: {inc_cap:.2f}")
print(f"Revolution capped at: {revol_cap:.2f}")

# 6. Applying Encoding on Purpose to group rare values as Others

purpose_counts = df_new['purpose'].value_counts()
rare_purpose = purpose_counts[purpose_counts<1000].index

df_new['purpose'] = df_new['purpose'].apply(
    lambda x: 'other' if x in rare_purpose else x
)

print(f"Purpose after rare encoding: {df_new['purpose'].value_counts()}")

# 7. One Hot Encoding for Categorical Features & Label Encoding for Target 

cat_cols = ['purpose','home_ownership', 'term']
df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)


def grade_to_tier_mapping(grade):
    if grade in ['A','B']:  # Low Risk
        return 0
    elif grade in ['C','D']:  # Medium Risk
        return 1
    else:
        return 2 # High Risk
    
df_new['encoded_grade'] = df_new['grade'].apply(grade_to_tier_mapping)

print(f'Risk Tier Distribution: {df_new['encoded_grade'].value_counts().sort_index()}')

print(f'Shape after Encoding: {df_new.shape}')

# 8. Feature Scaling

num_features = ['loan_amnt', 'annual_inc', 'dti',
                'fico_avg', 'revol_bal', 'revol_util', 
                'total_acc', 'emp_length']

ss= StandardScaler()
df_new[num_features] = ss.fit_transform(df_new[num_features]) 


# 9. Applying Train Test Split

X = df_new.drop(['grade','encoded_grade'], axis=1)
y= df_new['encoded_grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify = y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Saving Processed Data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)