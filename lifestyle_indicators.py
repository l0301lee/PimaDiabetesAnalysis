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

# Impact of Lifestyle Indicators
# Control for blood pressure and skin thickness
X = df[['BMI', 'BloodPressure', 'SkinThickness']]
X = sm.add_constant(X)
y = df['Outcome']
logit_model_lifestyle = sm.Logit(y, X)
result_lifestyle = logit_model_lifestyle.fit()

# Save the summary to a text file
with open('results/lifestyle_indicator_summary.txt', 'w') as f:
    f.write(result_lifestyle.summary().as_text())

# Visualizing
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))

# Histograms and Boxplots
for i, column in enumerate(['BMI', 'BloodPressure', 'SkinThickness']):
    # Histograms
    sns.histplot(data=df, x=column, kde=True, ax=axes[i][0])
    axes[i][0].set_title(f'Distribution of {column}')

    # Boxplots
    sns.boxplot(data=df, x='Outcome', y=column, ax=axes[i][1])
    axes[i][1].set_title(f'{column} by Outcome')

plt.tight_layout()
plt.savefig('results/distribution_and_boxplots.png')

# Function to plot and save predicted probabilities
def plot_predicted_probabilities(df_sorted, variable, model, filename):
    predicted_probs = model.predict(sm.add_constant(df_sorted[[variable, 'BloodPressure', 'SkinThickness']]))
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted[variable], predicted_probs, '-')
    plt.title(f'Predicted Probabilities of Diabetes by {variable}')
    plt.xlabel(variable)
    plt.ylabel('Probability of Diabetes')
    plt.savefig(f'results/{filename}')
    plt.close()

# Plotting predicted probabilities for BMI, BloodPressure, and SkinThickness
plot_predicted_probabilities(df.sort_values(by="BMI"), "BMI", result_lifestyle, 'predicted_probabilities_bmi.png')
plot_predicted_probabilities(df.sort_values(by="BloodPressure"), "BloodPressure", result_lifestyle, 'predicted_probabilities_bp.png')
plot_predicted_probabilities(df.sort_values(by="SkinThickness"), "SkinThickness", result_lifestyle, 'predicted_probabilities_st.png')
