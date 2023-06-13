# Learn a decision tree and include functions for producing counterfactual
# experiments from a decision tree and dataset.
from __future__ import annotations
import torch
from torch import nn, sigmoid


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