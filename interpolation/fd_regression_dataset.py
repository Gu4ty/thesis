import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np


class FdRegressionDataset(Dataset):
    def __init__(
        self,
        t,
        sol,
        eqn_index,
        order,
        normalization_factor,
        include_time=True,
        max_time=1,
        transform=None,
        target_transform=None,
    ):
        self.X = []
        self.Y = []
        var_sol = sol[:, eqn_index]
        if order == 1:
            deriv = np.diff(var_sol) / np.diff(t)
        else:
            deriv = np.gradient(var_sol, t, edge_order=2)

        for (i, d) in enumerate(deriv):
            x = [t[i] / max_time] if include_time else []
            sol_i = [sol[i, j] / normalization_factor for j in range(len(sol[0, :]))]
            x += sol_i
            self.X.append(x)
            self.Y.append([deriv[i]])

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
