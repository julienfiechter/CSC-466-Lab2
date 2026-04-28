import json
import numpy as np
import pandas as pd

def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


def categorical_info_gain(X_col, y):
    total_entropy = entropy(y)

    values, counts = np.unique(X_col, return_counts=True)

    weighted_entropy = 0
    for v, count in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (count / len(y)) * entropy(subset_y)

    return total_entropy - weighted_entropy

def categorical_gain_ratio(X_col, y):
    ig = categorical_info_gain(X_col, y)
    split_ent = entropy(X_col)

    if split_ent == 0:
        return 0

    return ig / split_ent

def numeric_info_gain(X_col, y, threshold):
    total_entropy = entropy(y)

    left_mask = X_col <= threshold
    right_mask = X_col > threshold

    y_left = y[left_mask]
    y_right = y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return 0

    weighted_entropy = (
        (len(y_left) / len(y)) * entropy(y_left) +
        (len(y_right) / len(y)) * entropy(y_right)
    )

    return total_entropy - weighted_entropy

def numeric_gain_ratio(X_col, y, threshold):
    ig = numeric_info_gain(X_col, y, threshold)

    left_mask = X_col <= threshold
    right_mask = X_col > threshold

    split_probs = np.array([left_mask.mean(), right_mask.mean()])
    split_probs = split_probs[split_probs > 0]

    split_info = -np.sum(split_probs * np.log2(split_probs))

    if split_info == 0:
        return 0

    return ig / split_info

def numeric_thresholds(X_col):
    unique_vals = np.sort(X_col.dropna().unique())

    thresholds = []
    for i in range(len(unique_vals) - 1):
        t = (unique_vals[i] + unique_vals[i + 1]) / 2
        thresholds.append(t)

    return thresholds

def make_leaf(y):
    counts = y.value_counts()
    decision = counts.idxmax()
    p = counts.max() / len(y)
    return {
        "decision": decision,
        "p": round(p, 3)
    }

def best_numeric_split(X_col, y, metric="info_gain"):
    best_threshold = None
    best_score = -1

    for threshold in numeric_thresholds(X_col):
        if metric == "info_gain":
            score = numeric_info_gain(X_col, y, threshold)
        elif metric == "gain_ratio":
            score = numeric_gain_ratio(X_col, y, threshold)
        else:
            raise ValueError("Invalid metric")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

def best_split(X, y, metric="info_gain"):
    best_attr = None
    best_score = -1
    best_threshold = None
    best_is_numeric = False

    for col in X.columns:
        if is_numeric(X[col]):
            threshold, score = best_numeric_split(X[col], y, metric)
            current_is_numeric = True
        else:
            threshold = None
            if metric == "info_gain":
                score = categorical_info_gain(X[col], y)
            elif metric == "gain_ratio":
                score = categorical_gain_ratio(X[col], y)
            else:
                raise ValueError("Invalid metric")
            current_is_numeric = False

        if score > best_score:
            best_score = score
            best_attr = col
            best_threshold = threshold
            best_is_numeric = current_is_numeric

    return best_attr, best_score, best_threshold, best_is_numeric


def build_tree(X, y, threshold=0.0, metric="info_gain"):
    if len(X.columns) == 0:
        return {"leaf": make_leaf(y)}

    if len(y.unique()) == 1:
        return {"leaf": make_leaf(y)}

    split_attr, split_gain, split_threshold, split_is_numeric = best_split(X, y, metric)

    if split_gain <= threshold or split_attr is None:
        return {"leaf": make_leaf(y)}

    tree = {
        "var": split_attr,
        "edges": []
    }

    if split_is_numeric:
        left_mask = X[split_attr] <= split_threshold
        right_mask = X[split_attr] > split_threshold

        branches = [
            ("<=", split_threshold, left_mask),
            (">", split_threshold, right_mask)
        ]

        for op, val, mask in branches:
            X_subset = X[mask]
            y_subset = y[mask]

            if len(y_subset) == 0:
                child = {"leaf": make_leaf(y)}
            else:
                child = build_tree(X_subset, y_subset, threshold, metric)

            edge = {
                "edge": {
                    "value": float(val),
                    "op": op
                }
            }

            if "leaf" in child:
                edge["edge"]["leaf"] = child["leaf"]
            else:
                edge["edge"]["node"] = child

            tree["edges"].append(edge)

    else:
        for val in X[split_attr].unique():
            mask = X[split_attr] == val
            X_subset = X[mask].drop(columns=[split_attr])
            y_subset = y[mask]

            if len(y_subset) == 0:
                child = {"leaf": make_leaf(y)}
            else:
                child = build_tree(X_subset, y_subset, threshold, metric)

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

        if "op" in edge:
            if edge["op"] == "<=" and row_value <= edge["value"]:
                if "leaf" in edge:
                    return edge["leaf"]["decision"]
                return predict_one(row, edge["node"])

            if edge["op"] == ">" and row_value > edge["value"]:
                if "leaf" in edge:
                    return edge["leaf"]["decision"]
                return predict_one(row, edge["node"])
        else:
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

        self.tree = build_tree(X, y, self.threshold, self.metric)
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