import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load the cleaned and preprocessed dataset
df = pd.read_csv('diabetes_clean.csv')

# Association Between Glucose Levels and Diabetes
# Apply logistic regression
X = df[['Glucose', 'Age']]
X = sm.add_constant(X)
y = df['Outcome']
logit_model_glucose = sm.Logit(y, X)
result_glucose = logit_model_glucose.fit()

# Save the summary to a text file
with open('results/logistic_regression_summary.txt', 'w') as f:
    f.write(result_glucose.summary().as_text())

# Generate a range of glucose values for predictions
glucose_range = np.linspace(X['Glucose'].min(), X['Glucose'].max(), 100)

# Hold Age constant at its mean value
age_constant = X['Age'].mean()

# Create a new dataframe for predictions
X_new = pd.DataFrame({
    'const': 1,
    'Glucose': glucose_range,
    'Age': age_constant
})

# Predict probabilities
y_pred = result_glucose.predict(X_new)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Glucose'], y, label='Data', alpha=0.5) # Scatter plot of actual data
plt.plot(glucose_range, y_pred, color='red', label='Logistic Regression Fit')  # Logistic regression curve

# Labeling the plot
plt.xlabel('Glucose Level')
plt.ylabel('Probability of Diabetes')
plt.title('Association between Glucose Levels and Diabetes')
plt.legend()

# Save the plot to a file
plt.savefig('results/glucose_diabetes_association.png')
