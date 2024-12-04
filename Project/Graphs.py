import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



# Get the path for the dataset.
file_path = 'Project\Data\modified_data_for_graphs.csv'


# Load the dataset.
df = pd.read_csv(file_path, delimiter = ';', na_values = ['Nr', 'NSP'])


# Get the missing values.
missing_values = df.isnull().sum()
print("\nNumber of missing values for each attribute:\n\n", missing_values)


# Get the duplicated values.
duplicate_instances = df.duplicated().sum()
print(f"\nNumber of duplicate instances: {duplicate_instances}.\n")


# Display unique values for each attribute.
for column in df.columns:
    unique_values = df[column].value_counts()
    print(f"\nAttribute {column}:\n", unique_values)


# Get the attributes.
columns_to_encode = [
    'Gender', 'Age', 'Breed', 'Number', 'Housing', 'Zone', 'Exterior Access',
    'Observations', 'Timide', 'Calme', 'Effrayé', 'Intelligent', 'Vigilant',
    'Persévérant', 'Affectueux', 'Amical', 'Solitaire', 'Brutal', 'Dominant',
    'Aggressive', 'Impulsive', 'Predictable', 'Distracted'
]


# Display the graphs.
for column in columns_to_encode:
    if column in df.columns:
        le = LabelEncoder()
        df[f'{column}_numeric'] = le.fit_transform(df[column].astype(str))
        counts = df[f'{column}_numeric'].value_counts()
        plt.figure(figsize = (6, 4))
        counts.plot(kind = 'bar', color = 'skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation = 0)
        plt.show()
    else:
        print(f"Column '{column}' not found in the dataset.")


# Get the numeric columns.
numeric_columns = df.select_dtypes(include = ['int64', 'float64']).columns


# Display the heatmap.
if not numeric_columns.empty:
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize = (12, 10))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
else:
    print("No numeric columns found for correlation analysis.")