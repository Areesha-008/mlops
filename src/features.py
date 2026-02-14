import pandas as pd
import os

os.makedirs("features", exist_ok=True)

df = pd.read_csv("data/processed/processed.csv")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df.to_csv("features/features.csv", index=False)

print("Feature engineering completed")
