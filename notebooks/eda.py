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

#Plot -3 Interest Rate by Grade
df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').astype(float)
plt.figure(figsize=(10,5))
sns.boxplot(x='grade',y='int_rate',data=df,
            order=['A','B','C','D','E','F'], palette='RdYlGn_r')
plt.title('Interest Rate by Grade')
plt.xlabel('Grade')
plt.ylabel('Interest Rate (%)')
plt.tight_layout()
plt.savefig('src/plots/03_Interest_rate_by_grade.png')
plt.show()
print("\n=== INTEREST RATE BY GRADE ===")
print(df.groupby('grade')['int_rate'].describe().round(2))

#Plot -4 Annual Income vs Loan Amount

income_cap= df['annual_inc'].quantile(0.99)
plot_df= df[df['annual_inc']<=income_cap]
plt.figure(figsize=(10,5))
sns.scatterplot(x='annual_inc',y='loan_amnt', hue='grade',
                data=plot_df.sample(1000), alpha=0.5,
                hue_order=['A','B','C','D','E','F'])
plt.title('Annual Income Vs Loan Amount by Grade')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.tight_layout()
plt.savefig('src/plots/04_income_vs_loanamt.png')
plt.show()
