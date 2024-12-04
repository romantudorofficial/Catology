import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset and consider 'Nr' and 'NSP' as missing values
file_path = 'Homework/Laboratory 3/data.csv'
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

# Step 2: Convert non-numeric columns into numeric values using label encoding
# Apply Label encoding for 'Gender', 'Breed', 'Age', 'Housing', and 'Zone'
le_gender = LabelEncoder()
df['Gender_numeric'] = le_gender.fit_transform(df['Gender'])

le_breed = LabelEncoder()
df['Breed_numeric'] = le_breed.fit_transform(df['Breed'])

le_age = LabelEncoder()
df['Age_numeric'] = le_age.fit_transform(df['Age'])

le_housing = LabelEncoder()
df['Housing_numeric'] = le_housing.fit_transform(df['Housing'])

le_zone = LabelEncoder()
df['Zone_numeric'] = le_zone.fit_transform(df['Zone'])

# Step 3: Plot the distribution of 'Gender' as a bar chart
gender_counts = df['Gender_numeric'].value_counts()
plt.figure(figsize=(6,4))
gender_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Step 4: Plot the distribution of 'Breed' as a bar chart
breed_counts = df['Breed_numeric'].value_counts()
plt.figure(figsize=(8,5))
breed_counts.plot(kind='bar', color='lightgreen')
plt.title('Distribution of Breed')
plt.xlabel('Breed')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Step 5: Plot the distribution of 'Age' as a bar chart
age_counts = df['Age_numeric'].value_counts()
plt.figure(figsize=(6,4))
age_counts.plot(kind='bar', color='orange')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Step 6: Plot the distribution of 'Housing' as a bar chart
housing_counts = df['Housing_numeric'].value_counts()
plt.figure(figsize=(6,4))
housing_counts.plot(kind='bar', color='purple')
plt.title('Distribution of Housing')
plt.xlabel('Housing')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Step 7: Plot the distribution of 'Zone' as a bar chart
zone_counts = df['Zone_numeric'].value_counts()
plt.figure(figsize=(6,4))
zone_counts.plot(kind='bar', color='red')
plt.title('Distribution of Zone')
plt.xlabel('Zone')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
