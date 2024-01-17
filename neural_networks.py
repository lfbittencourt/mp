import numpy as np
import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor


class Module(nn.Module):
    def __init__(self, input_dimensions, dropout_rate=0):
        super(Module, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(input_dimensions, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, X):
        if X.dtype != torch.float32:
            X = X.to(torch.float32)

        X = self.module(X)

        return X


class NeuralNet(NeuralNetRegressor):
    def fit(self, X, y):
        # Sets the input dimensions of the module according to current X
        self.set_params(module__input_dimensions=X.shape[1])

        # Check if X is a Pandas DataFrame and convert it to a numpy array
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        # Check if y is a Pandas Series and convert it to a numpy array
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if y.dtype != np.float32:
            y = y.astype(np.float32)

        # Reshape y to 2D if it is 1D
        # From https://github.com/skorch-dev/skorch/issues/701#issuecomment-700943377
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return super().fit(X, y)
