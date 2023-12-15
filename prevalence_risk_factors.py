import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load the cleaned and preprocessed dataset
df = pd.read_csv('diabetes_clean.csv')

# Prevalence and Risk Factors by Age Group
# Group the data by age group and calculate prevalence
age_groups = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60])
prevalence_by_age = df.groupby(age_groups)['Outcome'].mean()

# Save prevalence by age group to txt file
prevalence_by_age.to_csv('results/prevalence_by_age.txt', header=True)

# Logistic Regression with Age and Risk Factors
X = df[['Pregnancies', 'BMI', 'Insulin', 'Age']].copy()
X['Age*Pregnancies'] = X['Age'] * X['Pregnancies']
X['Age*BMI'] = X['Age'] * X['BMI']
X['Age*Insulin'] = X['Age'] * X['Insulin']
y = df['Outcome']

# Fit logistic regression
logit_model_sklearn = LogisticRegression(max_iter=10000)  # Increased max_iter for convergence
logit_model_sklearn.fit(X, y)

# Cross-validation
results = cross_validate(logit_model_sklearn, X, y, cv=5)
cross_val_result = "Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (results['test_score'].mean(), results['test_score'].std() * 2)
print(cross_val_result)

# Save cross-validation results to txt file
with open('results/cross_validation_results.txt', 'w') as f:
    f.write(cross_val_result)

# Logistic regression model using statsmodels:
X = sm.add_constant(X)
logit_model_statsmodels = sm.Logit(y, X)
result = logit_model_statsmodels.fit()
print(result.summary())

with open('results/prevalence_risk_factor_summary.txt', 'w') as f:
    f.write(result.summary().as_text())

# Create bar chart for prevalence by age group
plt.figure(figsize=(10, 6))
prevalence_by_age.plot(kind='bar', color='skyblue')
plt.title("Diabetes Prevalence by Age Group")
plt.ylabel("Prevalence")
plt.xlabel("Age Group")
plt.xticks(rotation=45)

# Save the figure
plt.savefig('results/prevalence_by_age_group.png')
