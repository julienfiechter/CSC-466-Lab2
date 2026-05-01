import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

def save_tree_visualization(X, y, threshold, output_file):
    model = create_model(X, threshold)
    model.fit(X, y)

    clf = model.named_steps["classifier"]

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        filled=True,
        rounded=True,
        class_names=clf.classes_,
        feature_names=None
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def create_model(X_train, threshold):
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            criterion="entropy",
            min_impurity_decrease=threshold,
            random_state=1))
    ])
    return model

def cross_validation(X, y, kfold, threshold):
    all_predictions = []
    all_true_predictions = []

    for i in range (len(kfold)):
        test = kfold[i]
        train = np.concatenate([kfold[j] for j in range(len(kfold)) if j != i])
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

        model = create_model(X_train, threshold)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_true_predictions.extend(y_test.tolist())

    accur = accuracy(all_true_predictions,all_predictions)
    confusion_mat = confusion_matrix(all_true_predictions, all_predictions)
    return accur, confusion_mat

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 crossValSKL.py <csv_file> <grid_file> [<output_tree_image>]")
        return
    filename = sys.argv[1]
    grid_file = sys.argv[2]
    output_tree = sys.argv[3] if len(sys.argv) > 3 else None
    X,y = load_input_file(filename)
    with open(grid_file, "r") as f:
        grid = json.load(f)
    thresholds = grid["InfoGain"]
    kfold = folds(len(y), k=10, seed=1)
    best_accuracy = -1
    best_threshold = None
    best_confusion_matrix = None

    for t in thresholds:
        accur, confusion_mat = cross_validation(X, y, kfold, t)
        if accur > best_accuracy:
            best_accuracy = accur
            best_threshold = t
            best_confusion_matrix = confusion_mat

    if output_tree:
        save_tree_visualization(X, y, best_threshold, output_tree)

    print(f"Splitting Metric: InfoGain, Threshold: {best_threshold}")
    print(f"Accuracy: {best_accuracy:.5f}")
    print(best_confusion_matrix)

if __name__ == "__main__":
    main()