import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from random import uniform


class RegressionDataset(Dataset):
    # Dataset for training:
    # N(t, y1(t), y2(t), ... , yn(t)) ~ f(t, y1(t), y2(t), ... , yn(t))
    # where f = y_i' for some i in [1,n]
    # X = (t, y1(t), y2(t), ... , yn(t))
    # Y = f(t, y1(t), y2(t), ... , yn(t))
    # y = [y1, y2, ..., yn]
    # y_prime = f
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
        self.X = []
        self.Y = []
        for _ in range(samples):
            t = uniform(interval_low, interval_high)
            yt = [yi(t) for yi in y]
            x = [t] + yt
            self.X.append(x)
            self.Y.append([y_prime(t)])

        self.X = Variable(torch.tensor(self.X).type(torch.float))
        self.Y = Variable(torch.tensor(self.Y).type(torch.float))

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


def y(t):
    return t * t


def y_p(t):
    return 2 * t


if __name__ == "__main__":
    dataset = RegressionDataset([y], y_p, 1, 3, 100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    x, y = next(iter(dataloader))
    print(f"X: {x} \n Y: {y}")
    # print(len(dataset))
    # print(dataset[1:10])
