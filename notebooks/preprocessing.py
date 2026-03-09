import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Data Loading

df = pd.read_csv(
    'data/accepted_2007_to_2018Q4.csv.gz',
    compression='gzip',
    low_memory=False,
    nrows=100000
)
print(f" Data loaded: {df.shape}")

# Selecting relevant features
features=[
    'loan_amnt', 'term', 'int_rate', 'annual_inc', 'purpose',
    'dti', 'fico_range_low', 'fico_range_high', 'open_acc',
    'revol_bal', 'revol_util', 'total_acc', 'home_ownership',
    'emp_length', 'grade'
]


df_new = df[features].copy()

print(f"Shape of new Dataframe: {df_new.shape}")
print(f"Missing values: {df_new.isnull().sum()}")

# Dropping redundant columns 

df_new['fico_avg'] = (df_new['fico_range_low'] + df_new['fico_range_high'])/2

df_new.drop(['fico_range_high','fico_range_low','open_acc'],axis=1,inplace=True)

print(f"Shape after dropping columns: {df_new.shape}")

# Handling Missing Values
missing_pct = (((df_new.isnull().sum())/len(df_new))*100).round(2)
print(f"Missing Percentages: ", missing_pct[missing_pct>0])

# Filling numerical columns with median
num_cols = ['loan_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'fico_avg']

for col in num_cols:
    df_new[col] = df_new[col].fillna(df_new[col].median())


#Filling actegorical columns with mode
cat_cols = ['term', 'purpose', 'home_ownership', 'emp_length']

for col in cat_cols:
    df_new[col] = df_new[col].fillna(df_new[col].mode()[0])


print(df_new.isnull().sum())    