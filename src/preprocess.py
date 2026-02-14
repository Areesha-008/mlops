import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/train.csv")

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

df.to_csv("data/processed/processed.csv", index=False)

print("Preprocessing completed")
