# Learn a decision tree and include functions for producing counterfactual
# experiments from a decision tree and dataset.
from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from src.source_target_dataset import SourceTargetDataset
import numpy as np

class DecisionTree:
    def __init__(self, max_depth: int) -> None:
        self._clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self._feature_names = None

    def fit(self, data: SourceTargetDataset, classnames: list[str]) -> None:
        self._clf.fit(data.samples(), data.labels())
        self._feature_names = classnames

    def argpartition(self, data: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, float]]]:
        def rec_partition(node: int, data: np.ndarray, feature: int, threshold: float) -> tuple[list[np.ndarray], list[tuple[int, float]]]:
            # This node is a leaf
            left = self._clf.tree_.children_left[node]
            right = self._clf.tree_.children_right[node]
            if left == right:
                return [data[:,-1]], [(feature, threshold)]
            feature = self._clf.tree_.feature[node]
            threshold = self._clf.tree_.threshold[node]
            part_l, part_r = self.split(data, feature, threshold)
            subl = rec_partition(left, part_l, -feature, threshold)
            subr = rec_partition(right, part_r, feature, threshold)
            return subl[0] + subr[0], subl[1] + subr[1]

        data = np.concatenate([data, np.arange(len(data))[...,None]], axis=1)
        return rec_partition(0, data, 0, 0)

    def visualize_tree(self) -> str:
        assert self._feature_names != None
        return tree.export_text(self._clf, decimals=7, feature_names=self._feature_names)

    def split(self, data: np.ndarray, feature: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        split_l = []
        split_r = []
        for sample in data:
            if sample[feature] <= threshold:
                split_l.append(sample)
            else:
                split_r.append(sample)
        return np.stack(split_l), np.stack(split_r)