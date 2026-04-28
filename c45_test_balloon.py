import json
import pandas as pd
from c45 import C45

balloon1 = pd.read_csv("adult+stretch.csv", skiprows=[1, 2])
balloon2 = pd.read_csv("adult-stretch.csv", skiprows=[1, 2])
balloon3 = pd.read_csv("yellow-small+adult-stretch.csv", skiprows=[1, 2])
balloon4 = pd.read_csv("yellow-small.csv", skiprows=[1, 2])

balloons = pd.concat([balloon1, balloon2, balloon3, balloon4], ignore_index=True)

X = balloons.drop(columns=["Inflated"])
y = balloons["Inflated"]

print("Gain Ratio Tree:")
model = C45(metric="gain_ratio", threshold=0.0)
tree = model.fit(X, y)
print(json.dumps(tree, indent=2))

print("Info Gain Tree:")
model_ig = C45(metric="info_gain", threshold=0.0)
tree_ig = model_ig.fit(X, y)
print(json.dumps(tree_ig, indent=2))

predictions = model.predict(X)
accuracy = sum(prediction == actual for prediction, actual in zip(predictions, y)) / len(y)
print("Training accuracy:", accuracy)