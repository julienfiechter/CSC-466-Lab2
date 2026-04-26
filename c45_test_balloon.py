import json
import pandas as pd
from c45 import C45, entropy, info_gain

balloon1 = pd.read_csv("adult+stretch.csv", skiprows=[1, 2])
balloon2 = pd.read_csv("adult-stretch.csv", skiprows=[1, 2])
balloon3 = pd.read_csv("yellow-small+adult-stretch.csv", skiprows=[1, 2])
balloon4 = pd.read_csv("yellow-small.csv", skiprows=[1, 2])

balloons = pd.concat([balloon1, balloon2, balloon3, balloon4], ignore_index=True)

X = balloons.drop(columns=["Inflated"])
y = balloons["Inflated"]

print("Entropy of target:", entropy(y))
for col in X.columns:
    print(f"Info Gain for {col}: {info_gain(X[col], y)}")

model = C45(threshold=0.0)
tree = model.fit(X, y)

print(json.dumps(tree, indent=2))

predictions = model.predict(X)
accuracy = sum(prediction == actual for prediction, actual in zip(predictions, y)) / len(y)
print("Training accuracy:", accuracy)