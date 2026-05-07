import sys
import json
import pandas as pd
from c45 import C45

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