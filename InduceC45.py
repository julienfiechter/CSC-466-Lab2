import sys
import json
import pandas as pd
from c45 import C45

def load_data(filename):
    meta = pd.read_csv(filename, nrows=3, header=None)

    class_col = meta.iloc[2].dropna().values[0]

    df = pd.read_csv(filename, skiprows=[1, 2])
    df.columns = df.columns.str.strip()

    X = df.drop(columns=[class_col])
    y = df[class_col]

    return X, y, class_col

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 InduceC45 <TrainingSetFile.csv> [<fileToSave>]")
        return

    filename = sys.argv[1]
    save_file = sys.argv[2] if len(sys.argv) > 2 else None

    X, y, class_col = load_data(filename)

    model = C45(metric="info_gain", threshold=0.0)
    tree = model.fit(X, y)

    output = {
        "dataset": filename,
        "node": tree
    }

    print(json.dumps(output, indent=2))

    if save_file:
        model.save_tree(save_file, filename)


if __name__ == "__main__":
    main()