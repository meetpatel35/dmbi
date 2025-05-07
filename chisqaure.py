import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load DataFrame from CSV file
df = pd.read_csv('heart_patient_data.csv')  # Update the path to your CSV file

# Data Visualization
fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # figure with 3 rows and 2 columns

# a. Bar Chart of Average Systolic Blood Pressure by Gender
sns.barplot(x='Gender', y='Blood Pressure (Systolic)', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Average Systolic Blood Pressure by Gender')
axs[0, 0].set_ylabel('Systolic Blood Pressure')
axs[0, 0].set_xlabel('Gender')

# b. Bar Chart of Average Cholesterol Level by Gender
sns.barplot(x='Gender', y='Cholesterol Level', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Average Cholesterol Level by Gender')
axs[0, 1].set_ylabel('Cholesterol Level')
axs[0, 1].set_xlabel('Gender')

# c. Histogram of BMI
sns.histplot(df['BMI'], bins=5, ax=axs[1, 0])
axs[1, 0].set_title('Histogram of BMI')

# d. Histogram of Heart Rate
sns.histplot(df['Heart Rate'], bins=5, ax=axs[1, 1])
axs[1, 1].set_title('Histogram of Heart Rate')

# e. Scatter Plot of Systolic Blood Pressure vs Cholesterol Level
sns.scatterplot(x='Blood Pressure (Systolic)', y='Cholesterol Level', data=df, ax=axs[2, 0])
axs[2, 0].set_title('Scatter Plot of Systolic Blood Pressure vs Cholesterol Level')
axs[2, 0].set_xlabel('Systolic Blood Pressure')
axs[2, 0].set_ylabel('Cholesterol Level')

# f. Q-Q Plot for BMI
stats.probplot(df['BMI'], dist="norm", plot=axs[2, 1])
axs[2, 1].set_title('Q-Q Plot for BMI')

plt.tight_layout()
plt.show()

# Statistical Data Analysis

# Chi-Square Test: Gender vs Heart Patient
contingency_table = pd.crosstab(df['Gender'], df['Heart Patient'])
chi2_statistic, p_value_chi2 = stats.chi2_contingency(contingency_table)[:2]
print(f"\nChi-Square Test Statistic: {chi2_statistic}")
print(f"Chi-Square Test p-value: {p_value_chi2}")

# Correlation Analysis: among numeric columns relevant
num_columns = ['Age', 'Blood Pressure (Systolic)', 'Blood Pressure (Diastolic)',
               'Cholesterol Level', 'Heart Rate', 'BMI']
correlation_matrix = df[num_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap for Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
