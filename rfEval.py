import sys
import json
import numpy as np
import pandas as pd
from randomForest import RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

import mlreport

def load_input_file(filename):
    preview = pd.read_csv(filename, nrows=3, header=None)
    row1 = preview.iloc[1].dropna().astype(str).str.strip()

    lab2_format = True
    for value in row1:
        try:
            int(value)
        except ValueError:
            lab2_format = False
            break

    if lab2_format:
        meta = preview
        class_col = meta.iloc[2].dropna().values[0].strip()

        df = pd.read_csv(filename, skiprows=[1, 2])
        df.columns = df.columns.str.strip()
        type_row = meta.iloc[1].tolist()
        drop_cols = []

        for col, col_type in zip(df.columns, type_row):
            try:
                if int(col_type) == -1:
                    drop_cols.append(col)
            except ValueError:
                pass

        df = df.drop(columns=drop_cols)

    else:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        class_col = df.columns[-1]

    X = df.drop(columns=[class_col])
    y = df[class_col]

    return X, y

def train_test_split_8020(X, y, seed=1):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)

    test_size = int(0.2 * len(y))

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)

    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    correct = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == predicted:
            correct += 1

    return correct / len(y_true)


def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    for actual, predicted in zip(y_true, y_pred):
        matrix.loc[actual, predicted] += 1

    return matrix

def load_grid(grid_file):
    with open(grid_file, "r") as f:
        raw = json.load(f)

    return {
        "threshold": raw["threshold"],
        "split": raw["split"],
        "num_trees": raw["NumTrees"],
        "num_attributes": raw["NumAttributes"],
        "num_data_points": raw["PercentData"]
    }


def split_name_to_metric(split_name):
    if split_name == "Information Gain":
        return "info_gain"
    elif split_name == "Information Gain Ratio":
        return "gain_ratio"
    else:
        raise ValueError(f"Invalid split metric: {split_name}")


def grid_search_rf(X_train, X_test, y_train, y_test, grid):
    best_model = None
    best_params = None
    best_accuracy = -1
    best_predictions = None
    best_confusion = None

    for split_name in grid["split"]:
        metric = split_name_to_metric(split_name)

        for threshold in grid["threshold"]:
            for num_trees in grid["num_trees"]:
                for num_attributes in grid["num_attributes"]:
                    for num_data_points in grid["num_data_points"]:

                        print(
                            f"Custom RF: split={split_name}, threshold={threshold}, "
                            f"trees={num_trees}, attrs={num_attributes}, data={num_data_points}",
                            flush=True
                        )

                        model = RandomForest(
                            num_attributes=num_attributes,
                            num_data_points=num_data_points,
                            num_trees=num_trees,
                            splitting_metric=metric,
                            threshold=threshold,
                            random_state=1
                        )

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        acc = accuracy(y_test, preds)

                        if acc > best_accuracy:
                            best_model = model
                            best_params = {
                                "split": split_name,
                                "threshold": threshold,
                                "NumTrees": num_trees,
                                "NumAttributes": num_attributes,
                                "PercentData": num_data_points
                            }
                            best_accuracy = acc
                            best_predictions = preds
                            best_confusion = confusion_matrix(y_test, preds)

    return best_model, best_params, best_accuracy, best_predictions, best_confusion

def create_sklearn_rf(X_train, num_trees, num_attributes, num_data_points):
    categorical_cols = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_cols = X_train.select_dtypes(
        exclude=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical_cols
            ),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=num_trees,
            max_features=num_attributes,
            max_samples=num_data_points,
            criterion="entropy",
            random_state=1
        ))
    ])

    return model


def grid_search_sklearn_rf(X_train, X_test, y_train, y_test, grid):
    best_model = None
    best_params = None
    best_accuracy = -1
    best_predictions = None
    best_confusion = None

    for num_trees in grid["num_trees"]:
        for num_attributes in grid["num_attributes"]:
            for num_data_points in grid["num_data_points"]:

                print(
                    f"Sklearn RF: trees={num_trees}, attrs={num_attributes}, data={num_data_points}",
                    flush=True
                )

                model = create_sklearn_rf(
                    X_train,
                    num_trees,
                    num_attributes,
                    num_data_points
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy(y_test, preds)

                if acc > best_accuracy:
                    best_model = model
                    best_params = {
                        "NumTrees": num_trees,
                        "NumAttributes": num_attributes,
                        "PercentData": num_data_points
                    }
                    best_accuracy = acc
                    best_predictions = preds
                    best_confusion = confusion_matrix(y_test, preds)

    return best_model, best_params, best_accuracy, best_predictions, best_confusion

def generate_report(
    output_file,
    custom_model,
    custom_params,
    custom_preds,
    skl_model,
    skl_params,
    skl_preds,
    X_train,
    X_test,
    y_train,
    y_test
):
    custom_report = mlreport.Report(
        model=custom_model,
        title="Custom Random Forest",
        description="Random Forest implementation using C4.5 decision trees.",
        model_type="classification",
        model_params=custom_params
    )

    custom_report.add_split("train", X_train, y_train)
    custom_report.add_split("test", X_test, y_test, y_pred=custom_preds)
    custom_report.build()

    skl_report = mlreport.Report(
        model=skl_model,
        title="Scikit-learn Random Forest",
        description="Scikit-learn RandomForestClassifier comparison model.",
        model_type="classification",
        model_params=skl_params
    )

    skl_report.add_split("train", X_train, y_train)
    skl_report.add_split("test", X_test, y_test, y_pred=skl_preds)
    skl_report.build()

    comparison = mlreport.ComparisonReport(
        reports=[custom_report, skl_report],
        title="Random Forest Comparison Report",
        description="Comparison of custom Random Forest and Scikit-learn Random Forest on the test split.",
        split="test"
    )

    comparison.build()
    comparison.to_html(output_file)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 rfEval.py <CSVFile> <outputfilename> [gridSearchSettings]")
        return

    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    grid_file = sys.argv[3] if len(sys.argv) > 3 else None

    X, y = load_input_file(csv_file)
    X_train, X_test, y_train, y_test = train_test_split_8020(X, y, seed=1)

    if grid_file is None:
        print("No grid file provided.")
        return

    grid = load_grid(grid_file)

    print("Running RandomForest grid search")
    custom_model, custom_params, custom_acc, custom_preds, custom_cm = grid_search_rf(
        X_train, X_test, y_train, y_test, grid
    )

    print("\nRunning sklearn RandomForest grid search")
    skl_model, skl_params, skl_acc, skl_preds, skl_cm = grid_search_sklearn_rf(
        X_train, X_test, y_train, y_test, grid
    )

    print("\n=== Custom RandomForest ===")
    print("Best params:", custom_params)
    print(f"Accuracy: {custom_acc:.5f}")
    print("Confusion matrix:")
    print(custom_cm)

    print("\n=== Scikit-learn RandomForest ===")
    print("Best params:", skl_params)
    print(f"Accuracy: {skl_acc:.5f}")
    print("Confusion matrix:")
    print(skl_cm)

    generate_report(
        output_file,
        custom_model,
        custom_params,
        custom_preds,
        skl_model,
        skl_params,
        skl_preds,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    main()