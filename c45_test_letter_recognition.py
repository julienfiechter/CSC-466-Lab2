import json
import pandas as pd
from c45 import C45

letter = pd.read_csv("letter-recognition.data.csv", skiprows=[1, 2])
X = letter.drop(columns=["lettr"])
y= letter["lettr"]

model = C45(metric="info_gain", threshold=0.0)
tree = model.fit(X, y)
print(json.dumps(tree, indent=2))

predictions = model.predict(X)
accuracy = sum(prediction == actual for prediction, actual in zip(predictions, y)) / len(y)
print("Training accuracy:", accuracy)

model.save_tree("test_tree_letter_recognition.json", "letter-recognition.data.csv")

