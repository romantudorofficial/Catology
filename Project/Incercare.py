import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset and consider 'Nr' and 'NSP' as missing values
file_path = r"C:\Users\iftim\Documents\GitHub\Catology\Project\data.csv"
# file_path = r"D:\1. University\2. Courses\1. Current Courses\01. Artificial Intelligence\3. GitHub Project\Project\data.csv"
df = pd.read_csv(file_path, delimiter=';', na_values=['Nr', 'NSP'])

# Step 1: Check for any missing values and identify duplicate rows
missing_values = df.isnull().sum()
duplicate_instances = df.duplicated().sum()

print("Missing values per column:\n", missing_values)
print("\nNumber of duplicate instances:", duplicate_instances)

# Display unique values for each column
for column in df.columns:
    unique_values = df[column].value_counts()
    print(f"\nAttribute {column}:\n", unique_values)

# Encode categorical columns
columns_to_encode = [
    'Gender', 'Age', 'Breed', 'Number', 'Housing', 'Zone', 'Exterior Access', 'Observations',
    'Timide', 'Calme', 'Effrayé', 'Intelligent', 'Vigilant', 'Persévérant', 'Affectueux',
    'Amical', 'Solitaire', 'Brutal', 'Dominant', 'Aggressive', 'Impulsive', 'Predictable', 'Distracted'
]

for column in columns_to_encode:
    if column in df.columns:
        le = LabelEncoder()
        df[f'{column}_numeric'] = le.fit_transform(df[column].astype(str))  # Convert to string if not already
        counts = df[f'{column}_numeric'].value_counts()

        # Plot the bar chart
        plt.figure(figsize=(6, 4))
        counts.plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.show()
    else:
        print(f"Column '{column}' not found in dataset.")

# Create a correlation matrix and heatmap for numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

if not numeric_columns.empty:
    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
else:
    print("No numeric columns found for correlation analysis.")
