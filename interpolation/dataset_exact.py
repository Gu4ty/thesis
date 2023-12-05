import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import random


class ExactDataset(Dataset):
    # ----------------------------------------------------------------------------
    # |             Dataset for training the neural network to interpolate F(t,y) given samples of F(t,y)
    # |     * Given:
    # |         - y: The vector y= (y1,y2,...,yn) of the solutions of the system
    # |         - f: The values of the right hand side F(t, y(t)) for each t and y
    # |         - samples : The number of points to interpolate in the interval
    # |     * Generates the tensors X and Y for train N(X) ~ F(t,y) = Y
    # |     * Assumptions:
    # |         - y(t) is normalized in t [0,1]
    # |         - f(t, [y1, y2, ...yn] ) t in [0,1]
    # ----------------------------------------------------------------------------
    def __init__(
        self,
        y,
        f,
        normalization_factor,
        samples=int(1e6),
        transform=None,
        target_transform=None,
    ):

        self.X = [[] for _ in range(samples)]
        self.Y = [[] for _ in range(samples)]
        for i in range(samples):
            self.X[i].append(random.uniform(0, 1))
            yf = []
            for yi in y:
                t = random.uniform(0, 1)
                self.X[i].append((yi(t)) / normalization_factor)
                yf.append(yi(t))

            self.Y[i] = [(f(self.X[i][0], yf))]

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
