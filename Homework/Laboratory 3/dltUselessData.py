import pandas as pd


df = pd.read_csv("data1.csv", delimiter=';')

df = df[~df['Breed'].isin(['NR', 'NSP'])]
print("Filtered DataFrame:")
print(df)

df.to_csv("filtered_data.csv", index=False)