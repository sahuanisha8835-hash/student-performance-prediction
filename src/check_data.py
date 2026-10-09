import pandas as pd

data = pd.read_csv("data/student_data.csv")

print("First 5 students:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())