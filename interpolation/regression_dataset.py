import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np


class RegressionDataset(Dataset):
    # ----------------------------------------------------------------------------
    # |             Dataset for training the neuroal network to interpolate F(t,y)
    # |     * Given:
    # |         - y: The vector y= (y1,y2,...,yn) of the solutions of the system
    # |         - y_prime: The values of the right hand side F(t, y(t)) for each t
    # |         - ineterval: The interval of interpolation [interval_low, interval_high]
    # |         - samples : The number of points to interpolate in the interval
    # |     * Generates the tensors X and Y for train N(X) ~ F(t,y) = Y
    # ----------------------------------------------------------------------------
    def __init__(
        self,
        y,
        y_prime,
        interval_low,
        interval_high,
        samples=int(1e6),
        transform=None,
        target_transform=None,
    ):

        self.X = [[] for _ in range(samples)]
        self.Y = [[] for _ in range(samples)]
        interval_samples = np.linspace(interval_low, interval_high, samples)
        for (i, t) in enumerate(interval_samples):
            yt = [float(yi(t)) / 1000 for yi in y]
            x = [float(t)] + yt
            yp = y_prime(t)

            self.X[i] = x
            self.Y[i] = [yp]

        self.X = Variable(torch.tensor(self.X))
        self.Y = Variable(torch.tensor(self.Y))

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
