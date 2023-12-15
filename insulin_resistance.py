import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load the cleaned and preprocessed dataset
df = pd.read_csv('diabetes_clean.csv')

# Insulin Resistance Investigation
# Control for BMI, blood pressure, and skin thickness
X = df[['Insulin', 'BMI', 'BloodPressure', 'SkinThickness']]
X = sm.add_constant(X)
y = df['Outcome']
logit_model_insulin = sm.Logit(y, X)
result_insulin = logit_model_insulin.fit()

# Save the summary to a text file
with open('results/insulin_resistance_summary.txt', 'w') as f:
    f.write(result_insulin.summary().as_text())

# Add a 'Probability' column to the DataFrame which holds the predictions
df['Probability'] = result_insulin.predict(X)

# Create a scatter plot with logistic curve
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Insulin', y='Probability', hue='Outcome', data=df)

# Sorting values for a smooth line
sorted_df = df.sort_values(by='Insulin')
plt.plot(sorted_df['Insulin'], sorted_df['Probability'], color='red')

# Add titles and labels
plt.title('Serum Insulin Level and Predicted Probability of Diabetes')
plt.xlabel('Serum Insulin Level')
plt.ylabel('Predicted Probability of Diabetes')
plt.legend(title='Observed Outcome')

# Save the plot
plt.savefig('results/insulin_probability_plot.png')
plt.close()  # Close the plot to avoid displaying it interactively