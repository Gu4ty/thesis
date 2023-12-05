import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np


class RegressionDataset(Dataset):
    # ----------------------------------------------------------------------------
    # |             Dataset for training the neural network to interpolate F(t,y)
    # |     * Given:
    # |         - y: The vector y= (y1,y2,..., yi,...,yn) of the solutions of the system
    # |         - y_prime: The values of the right hand side F(t, y(t)) for each t
    # |         - normalization_factor: To scale the output of the functions yi into [0,1]
    # |         - samples : The number of points to interpolate in the interval
    # |     * Generates the tensors X and Y for train N(X) ~ F(t,y) = Y
    # |     * Also computes and saves the means of the values of X in X_means
    # |     * Assumptions:
    # |         - The functions yi and y_prime are normalized in time [0,1]
    # |         - The interval of time is always [0,1] because of the normalization of the functions
    # ----------------------------------------------------------------------------
    def __init__(
        self,
        y,
        y_prime,
        normalization_factor,
        samples=int(1e6),
        include_time=True,
        interval_low=0,
        interval_high=1,
        transform=None,
        target_transform=None,
    ):
        self.X_means = [0 for _ in range(len(y) + 1)]
        if not include_time:
            self.X_means = [0 for _ in range(len(y))]
        self.X = [[] for _ in range(samples)]
        self.Y = [[] for _ in range(samples)]

        interval_samples = np.linspace(interval_low, interval_high, samples)
        for (i, t) in enumerate(interval_samples):
            yt = [yi(t) for yi in y]
            yt_norm = [yi / normalization_factor for yi in yt]
            if include_time:
                x = [t] + yt
                x_norm = [t] + yt_norm
            else:
                x = yt
                x_norm = yt_norm
            yp = y_prime(t)

            for (idx, xi) in enumerate(x):
                self.X_means[idx] += xi
            self.X[i] = x_norm
            self.Y[i] = [yp]

        self.X_means = [xi / samples for xi in self.X_means]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.X = Variable(torch.tensor(self.X, dtype=torch.double).to(device))
            self.Y = Variable(torch.tensor(self.Y, dtype=torch.double).to(device))
        else:
            self.X = Variable(torch.tensor(self.X, dtype=torch.double))
            self.Y = Variable(torch.tensor(self.Y, dtype=torch.double))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def concat_dataset(self, dataset):
        self.X = torch.cat((self.X, dataset.X), 0)  # type: ignore
        self.Y = torch.cat((self.Y, dataset.Y), 0)  # type: ignore
        self.X = Variable(self.X)
        self.Y = Variable(self.Y)
