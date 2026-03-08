import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Display Settings
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

#Loading Data
df= pd.read_csv(
    'data/accepted_2007_to_2018Q4.csv.gz',
    compression='gzip',
    low_memory=False,
    nrows=100000
)

print(f"Data loaded! {df.shape}")

# Plot-1 Target Variable Distribution
plt.figure(figsize=(10,5))
grade_counts= df['grade'].value_counts().sort_index()
sns.barplot(x=grade_counts.index, y=grade_counts.values, palette= 'Blues_d')
plt.title('Grade Distribution')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('src/plots/01_grade_distribution.png')
plt.show()
print("\n=== GRADE DISTRIBUTION ===")
print(grade_counts)
print(f"\n Grade %: \n {(grade_counts/len(df)*100).round(2)}")


#Plot- 2 Loan Amount Distribution
plt.figure(figsize=(10,5))
sns.histplot(df['loan_amnt'],bins=50, color='steelblue',kde=True)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('src/plots/02_loan_amount_distribution.png')
plt.show()
print("\n=== LOAN AMOUNT STATS ===")
print(df['loan_amnt'].describe())
