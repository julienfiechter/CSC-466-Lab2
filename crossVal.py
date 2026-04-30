import json
import sys
import numpy as np
import pandas as pd
from c45 import C45

def load_input_file(filename):
    meta = pd.read_csv(filename, nrows=3, header=None)

    class_col = meta.iloc[2].dropna().values[0].strip()

    df = pd.read_csv(filename, skiprows=[1, 2])
    df.columns = df.columns.str.strip()

    X = df.drop(columns=[class_col])
    y = df[class_col]

    return X, y

def folds(n, k = 10, seed=1):
    indices = np.arange(n)
    np.random.seed(seed)
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

def cross_validation(X, y, kfold, metric, threshold ):
    all_predictions = []
    all_true_predictions = []

    for i in range (len(kfold)):
        test = kfold[i]
        train = np.concatenate([kfold[j] for j in range(len(kfold)) if j != i])
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

        model = C45(metric = metric, threshold = threshold)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_true_predictions.extend(y_test.tolist())

    accur = accuracy(all_true_predictions,all_predictions)
    confusion_mat = confusion_matrix(all_true_predictions, all_predictions)
    return accur, confusion_mat

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 crossVal.py <csv_file> <grid_file> [<output_tree.json>]")
        return
    filename = sys.argv[1]
    grid_file = sys.argv[2]
    output_tree = sys.argv[3] if len(sys.argv) > 3 else None
    X,y = load_input_file(filename)

    with open(grid_file, "r") as f:
        grid = json.load(f)
    metric_map = {
        "InfoGain": "info_gain",
        "Ratio": "gain_ratio"
    }
    kfold = folds(len(y), k=10, seed=1)
    best_accuracy = -1
    best_params = None
    best_confusion_matrix = None

    for key in grid:
        for threshold in grid[key]:
            metric = metric_map[key]
            accur, confusion_mat = cross_validation(X, y, kfold, metric, threshold)
            if accur > best_accuracy:
                best_accuracy = accur
                best_params = (key, threshold)
                best_confusion_matrix = confusion_mat

    if output_tree:
        metric = metric_map[best_params[0]]
        threshold = best_params[1]
        model = C45(metric=metric, threshold=threshold)
        model.fit(X, y)
        model.save_tree(output_tree)

    print(f"Splitting Metric: {best_params[0]}, Threshold: {best_params[1]}")
    print(f"Accuracy: {best_accuracy:.5f}")
    print(best_confusion_matrix)

if __name__ == "__main__":
    main()