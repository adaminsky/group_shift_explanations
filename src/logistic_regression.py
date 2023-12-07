# Learn a decision tree and include functions for producing counterfactual
# experiments from a decision tree and dataset.
from __future__ import annotations
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from src.source_target_dataset import SourceTargetDataset
import numpy as np
import torch
from torch import nn, sigmoid


class Logistic_regression:
    def __init__(self) -> None:
        self._clf = LogisticRegression()
        # self._clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self._feature_names = None

    def fit(self, data: SourceTargetDataset, classnames: list[str]) -> None:
        fit_model = self._clf.fit(data.samples(), data.labels())
        self._feature_names = classnames
        return fit_model

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


class PTLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PTLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class PTNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PTNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc2(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc3(x)))
        x = self.fc4(x)
        outputs = torch.sigmoid(x)
        return outputs

    def embed(self, x):
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc2(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc3(x)))
        outputs = x
        return outputs


class PTNNSimple(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PTNNSimple, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        outputs = x
        # outputs = torch.sigmoid(x)
        return outputs

class FFNetwork(nn.Module):
    def __init__(self, input_size):
        super(FFNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        out = sigmoid(out)
        return out