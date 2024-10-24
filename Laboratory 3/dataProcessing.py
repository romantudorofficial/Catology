import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset and consider 'Nr' and 'NSP' as missing values
file_path = 'data.csv'
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

# Encode 'Age', 'Housing', and 'Zone' columns
le_age = LabelEncoder()
df['Age_numeric'] = le_age.fit_transform(df['Age'])

le_housing = LabelEncoder()
df['Housing_numeric'] = le_housing.fit_transform(df['Housing'])

le_zone = LabelEncoder()
df['Zone_numeric'] = le_zone.fit_transform(df['Zone'])

# Step 3: Plot the distributions in a single figure with multiple subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Plot the distribution of 'Gender'
gender_counts = df['Gender_numeric'].value_counts()
axs[0, 0].bar(gender_counts.index, gender_counts.values, color='skyblue')
axs[0, 0].set_title('Distribution of Gender')
axs[0, 0].set_xlabel('Gender')
axs[0, 0].set_ylabel('Frequency')
# Fix mismatch between tick positions and labels
axs[0, 0].set_xticks(gender_counts.index)
axs[0, 0].set_xticklabels(le_gender.inverse_transform(gender_counts.index))

# Plot the distribution of 'Breed'
breed_counts = df['Breed_numeric'].value_counts()
axs[0, 1].bar(breed_counts.index, breed_counts.values, color='lightgreen')
axs[0, 1].set_title('Distribution of Breed')
axs[0, 1].set_xlabel('Breed')
axs[0, 1].set_ylabel('Frequency')
# Fix mismatch between tick positions and labels
axs[0, 1].set_xticks(breed_counts.index)
axs[0, 1].set_xticklabels(le_breed.inverse_transform(breed_counts.index), rotation=45, ha='right')

# Plot the distribution of 'Age'
age_counts = df['Age_numeric'].value_counts()
axs[1, 0].bar(age_counts.index, age_counts.values, color='orange')
axs[1, 0].set_title('Distribution of Age')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_xticks(age_counts.index)
axs[1, 0].set_xticklabels(le_age.inverse_transform(age_counts.index))

# Plot the distribution of 'Housing'
housing_counts = df['Housing_numeric'].value_counts()
axs[1, 1].bar(housing_counts.index, housing_counts.values, color='purple')
axs[1, 1].set_title('Distribution of Housing')
axs[1, 1].set_xlabel('Housing')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].set_xticks(housing_counts.index)
axs[1, 1].set_xticklabels(le_housing.inverse_transform(housing_counts.index))

# Plot the distribution of 'Zone'
zone_counts = df['Zone_numeric'].value_counts()
axs[2, 0].bar(zone_counts.index, zone_counts.values, color='red')
axs[2, 0].set_title('Distribution of Zone')
axs[2, 0].set_xlabel('Zone')
axs[2, 0].set_ylabel('Frequency')
axs[2, 0].set_xticks(zone_counts.index)
axs[2, 0].set_xticklabels(le_zone.inverse_transform(zone_counts.index))

# Hide the unused subplot (bottom-right corner)
fig.delaxes(axs[2, 1])

# Adjust layout
plt.tight_layout()

# Show all plots at once
plt.show()
