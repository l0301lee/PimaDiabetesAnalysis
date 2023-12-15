import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('diabetes.csv')  # replace 'diabetes.csv' with the path to your data file

# Step 2: Inspect the dataset for any obvious issues
print(df.head())
print(df.describe())
print(df.info())

# Step 3: Handle Missing Values
# If the dataset uses a placeholder (like 0) for missing values, replace them with NaN
# For example, a blood pressure of 0 is not feasible and likely represents a missing value.
# Replace 0 with NaN in such columns.
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Check for missing values
print(df.isnull().sum())

# Option 1: Drop rows with missing values (if the dataset is large and the number of missing values is small)
df = df.dropna()

# Step 4: Handle Outliers
# For this step, define a function to find and remove outliers using IQR or z-score methods.
def remove_outliers(df, column_list):
    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Apply the function to your dataset (you might want to keep the original indices)
df_clean = remove_outliers(df, columns_with_zeros)

# Step 7: Export the Cleaned Dataset
df_clean.to_csv('diabetes_clean.csv', index=False)