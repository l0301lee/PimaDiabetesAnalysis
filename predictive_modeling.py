import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Load the cleaned and preprocessed dataset
df = pd.read_csv('diabetes_clean.csv')

# Predictive Modeling
# Split the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
report = classification_report(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print(report)

# Save the evaluation metrics to txt file
with open('results/evaluation_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"ROC AUC Score: {roc_auc}\n\n")
    f.write(f"Classification Report:\n{report}\n")

# Feature importance
importances = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)

# Save feature importances to txt file
with open('results/feature_importances.txt', 'w') as f:
    for i in importances.importances_mean.argsort()[::-1]:
        f.write(f"{X.columns[i]}: {importances.importances_mean[i]}\n")

# Extract feature importances
importance_values = importances.importances_mean
features = X.columns

# Sort feature importances in descending order and get the indices
sorted_idx = np.argsort(importance_values)[::-1]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_values[sorted_idx], y=features[sorted_idx], palette="viridis")
plt.title('Feature Importance using Permutation Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')

# Save the figure
plt.savefig('results/feature_importance.png')
