import json
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

def load_input_file(filename):
    df = pd.read_csv(filename)
    label = df.columns[-1]
    X = df.drop(columns=[label])
    y = df[label]
    return X, y

def folds(n, k = 10):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return np.array_split(indices, k)

def accuracy(y_true, y_pred):
    correct = 0
    for actual, predicted in zip(y_true, y_pred):
        if predicted == actual:
            correct += 1
    return correct/ len(y_true)

def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(predictions for predictions in y_pred if predictions is not None))
    matrix = pd.DataFrame(0, index = labels, columns = labels)
    for actual, predicted in zip(y_true, y_pred):
        if predicted is not None:
            matrix.loc[actual, predicted] += 1
    return matrix

def create_model(threshold):
   return Pipeline([
       ("encoder", OrdinalEncoder(
           handle_unknown="use_encoded_value", unknown_value=-1
       )),
       ("classifier", DecisionTreeClassifier(
           criterion="entropy",
           min_impurity_decrease= threshold
       ))
   ])

def cross_validation(X, y, kfold, threshold ):
    all_predictions = []
    all_true_predictions = []

    for i in range (len(kfold)):
        test = kfold[i]
        train = np.concatenate([kfold[j] for j in range(len(kfold)) if j != i])
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

        model = create_model(threshold)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_true_predictions.extend(y_test.tolist())

    accur = accuracy(all_true_predictions,all_predictions)
    confusion_mat = confusion_matrix(all_true_predictions, all_predictions)
    return accur, confusion_mat

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 crossValSKL.py <csv_file> <grid_file>")
        return
    filename = sys.argv[1]
    grid_file = sys.argv[2]
    X,y = load_input_file(filename)
    with open(grid_file, "r") as f:
        grid = json.load(f)
    thresholds = grid["InfoGain"]
    kfold = folds(len(y), k=10)
    best_accuracy = -1
    best_threshold = None
    best_confusion_matrix = None

    for t in thresholds:
        accur, confusion_mat = cross_validation(X, y, kfold, t)
        if accur > best_accuracy:
            best_accuracy = accur
            best_threshold = t
            best_confusion_matrix = confusion_mat

    print(f"Splitting Metric: InfoGain, Threshold: {best_threshold}")
    print(f"Accuracy: {best_accuracy:.5f}")
    print(best_confusion_matrix)

if __name__ == "__main__":
    main()