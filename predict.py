import sys
import pandas as pd
from c45 import C45

def load_data(filename):
    meta = pd.read_csv(filename, nrows=3, header=None)

    class_col = meta.iloc[2].dropna().values[0].strip()

    df = pd.read_csv(filename, skiprows=[1, 2])
    df.columns = df.columns.str.strip()

    X = df.drop(columns=[class_col])
    y = df[class_col]

    return X, y, class_col


def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    for actual, predicted in zip(y_true, y_pred):
        matrix.loc[actual, predicted] += 1

    return matrix


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py <CSVFile> <JSONFile> [eval]")
        return

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    eval_mode = len(sys.argv) > 3 and sys.argv[3] == "eval"

    X, y, class_col = load_data(csv_file)

    model = C45()
    model.read_tree(json_file)

    predictions = model.predict(X)

    for pred in predictions:
        print(pred)

    if eval_mode:
        total = len(y)
        correct = sum(pred == actual for pred, actual in zip(predictions, y))
        incorrect = total - correct
        accuracy = correct / total
        error_rate = incorrect / total

        cm = confusion_matrix(y, predictions)

        print()
        print(f"Total number of records classified: {total}")
        print(f"Total number of records correctly classified: {correct}")
        print(f"Total number of records incorrectly classified: {incorrect}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Error rate: {error_rate:.4f}")
        print("Confusion matrix:")
        print(cm)


if __name__ == "__main__":
    main()