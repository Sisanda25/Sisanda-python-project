import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./cleaned.csv')

# Data Handling
# Check for duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates if any
if duplicates > 0:
    data = data.drop_duplicates()
    print(f"Duplicates removed. New data shape: {data.shape}")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Handle missing values (example: fill with mean for numerical columns)
for column in data.select_dtypes(include=[np.number]).columns:
    data[column].fillna(data[column].mean(), inplace=True)

# Data Exploration
# Display the first few rows
print("First few rows of the data:")
print(data.head())

# Basic information
print("Data Info:")
print(data.info())

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Visualize distributions of numerical features
sns.set(style="whitegrid")
numerical_columns = data.select_dtypes(include=[np.number]).columns
for column in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Example analysis: group by a categorical variable
if 'category_column' in data.columns:  # Replace with actual categorical column name
    group_data = data.groupby('category_column').mean()
    print("Grouped data by category_column:")
    print(group_data)

# Visualization: Comparing two variables
if 'variable1' in data.columns and 'variable2' in data.columns:  # Replace with actual column names
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=data, x='variable1', y='variable2')
    plt.title('Scatter plot of variable1 vs variable2')
    plt.xlabel('variable1')
    plt.ylabel('variable2')
    plt.show()

