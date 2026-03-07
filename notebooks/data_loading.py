import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

df= pd.read_csv('../data/accepted_2007_to_2018Q4.csv.gz',
                compression='gzip',
                low_memory=False,
                nrows=100000
                )


print(f"Shape: {df.shape}")
print(f"\nNumber of columns: {len(df.columns)}")
print(f"\nFirst few columns: {df.columns[:10].tolist()}")
print(f"\nMissing values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
