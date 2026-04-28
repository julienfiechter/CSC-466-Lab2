import json
import pandas as pd
from c45 import C45

nursery = pd.read_csv("nursery.csv", skiprows=[1, 2])
nursery = nursery[nursery["class"] != "recommended"]
X = nursery.drop(columns=["class"])
y= nursery["class"]

model = C45(metric="gain_ratio", threshold=0.0)
tree = model.fit(X, y)
print(json.dumps(tree, indent=2))

predictions = model.predict(X)
accuracy = sum(prediction == actual for prediction, actual in zip(predictions, y)) / len(y)
print("Training accuracy:", accuracy)

model.save_tree("test_tree_nursery.json", "nursery.csv")

