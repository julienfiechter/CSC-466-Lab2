import json
import numpy as np
import pandas as pd


def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


def info_gain(X_col, y):
    total_entropy = entropy(y)

    values, counts = np.unique(X_col, return_counts=True)

    weighted_entropy = 0
    for v, count in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (count / len(y)) * entropy(subset_y)

    return total_entropy - weighted_entropy


def make_leaf(y):
    counts = y.value_counts()
    decision = counts.idxmax()
    p = counts.max() / len(y)
    return {
        "decision": decision,
        "p": round(p, 3)
    }


def best_split(X, y):
    best_attr = None
    best_gain = -1

    for col in X.columns:
        gain = info_gain(X[col], y)
        if gain > best_gain:
            best_gain = gain
            best_attr = col

    return best_attr, best_gain


def build_tree(X, y, threshold=0.0):
    if len(X.columns) == 0:
        return {"leaf": make_leaf(y)}

    if len(y.unique()) == 1:
        return {"leaf": make_leaf(y)}

    split_attr, split_gain = best_split(X, y)

    if split_gain <= threshold:
        return {"leaf": make_leaf(y)}

    tree = {
        "var": split_attr,
        "edges": []
    }

    for val in X[split_attr].unique():
        mask = X[split_attr] == val
        X_subset = X[mask].drop(columns=[split_attr])
        y_subset = y[mask]

        if len(y_subset) == 0:
            child = {"leaf": make_leaf(y)}
        else:
            child = build_tree(X_subset, y_subset, threshold)

        edge = {
            "edge": {
                "value": val
            }
        }

        if "leaf" in child:
            edge["edge"]["leaf"] = child["leaf"]
        else:
            edge["edge"]["node"] = child

        tree["edges"].append(edge)

    return tree


def predict_one(row, tree):
    if "leaf" in tree:
        return tree["leaf"]["decision"]

    split_attr = tree["var"]
    row_value = row[split_attr]

    for edge_obj in tree["edges"]:
        edge = edge_obj["edge"]
        if edge["value"] == row_value:
            if "leaf" in edge:
                return edge["leaf"]["decision"]
            return predict_one(row, edge["node"])

    return None


class C45:
    def __init__(self, metric="info_gain", threshold=0.0):
        self.metric = metric
        self.threshold = threshold
        self.tree = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.tree = build_tree(X, y, self.threshold)
        return self.tree

    def predict(self, X):
        if self.tree is None:
            raise ValueError("Model has not been trained yet.")

        predictions = []
        for _, row in X.iterrows():
            predictions.append(predict_one(row, self.tree))
        return predictions

    def save_tree(self, filename, dataset_name=None):
        if self.tree is None:
            raise ValueError("No tree to save.")

        output = {
            "dataset": dataset_name if dataset_name is not None else "",
            "node": self.tree
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

    def read_tree(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        self.tree = data["node"]
        return self.tree