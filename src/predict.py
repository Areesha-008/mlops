import pandas as pd
import pickle
import os

os.makedirs("results", exist_ok=True)

model = pickle.load(open("models/model.pkl", "rb"))

df = pd.read_csv("features/features.csv")

X = df.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1, errors='ignore')

predictions = model.predict(X)

pd.DataFrame(predictions, columns=["Predictions"]).to_csv("results/predictions.csv", index=False)

print("Predictions generated")
