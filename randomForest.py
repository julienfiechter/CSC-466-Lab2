import pandas as pd
import numpy as np
from collections import Counter
from c45 import C45


class RandomForest:
    def __init__(
        self,
        num_attributes,
        num_data_points,
        num_trees,
        splitting_metric="info_gain",
        threshold=0.0,
        random_state=42
    ):
        self.num_attributes = num_attributes
        self.num_data_points = num_data_points
        self.num_trees = num_trees
        self.splitting_metric = splitting_metric
        self.threshold = threshold
        self.random_state = random_state

        self.trees = []
        self.tree_attributes = []

    def get_samp_size(self, n_rows):
        if 0 < self.num_data_points <= 1:
            return int(self.num_data_points * n_rows)
        return int(self.num_data_points)

    def majority_vote(self, predictions):
        counts = Counter(predictions)
        max_count = max(counts.values())

        tied = [label for label, count in counts.items() if count == max_count]

        return sorted(tied)[0]

    def fit(self, X, y):
        self.trees = []
        self.tree_attributes = []

        rng = np.random.default_rng(self.random_state)

        n_rows = len(X)
        all_attributes = X.columns.tolist()

        sample_size = self.get_samp_size(n_rows)
        sample_size = max(1, min(sample_size, n_rows))

        attr_count = min(self.num_attributes, len(all_attributes))

        for _ in range(self.num_trees):
            row_indices = rng.choice(n_rows, size=sample_size, replace=True)

            selected_attrs = rng.choice(
                all_attributes,
                size=attr_count,
                replace=False
            ).tolist()

            X_sample = X.iloc[row_indices][selected_attrs]
            y_sample = y.iloc[row_indices]

            tree = C45(
                metric=self.splitting_metric,
                threshold=self.threshold
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)
            self.tree_attributes.append(selected_attrs)

        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("RandomForest has not been trained yet.")

        final_predictions = []

        for _, row in X.iterrows():
            tree_votes = []

            for tree, attrs in zip(self.trees, self.tree_attributes):
                row_df = pd.DataFrame([row[attrs]])
                pred = tree.predict(row_df)[0]
                tree_votes.append(pred)

            final_predictions.append(self.majority_vote(tree_votes))

        return final_predictions