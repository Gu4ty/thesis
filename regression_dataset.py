import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


class RegressionDataset(Dataset):
    # Regression for function from Rn to R
    # x = [x1,x2,...,xn]
    # y \in R
    def __init__(self, x_samples, y_samples, transform=None, target_transform=None):
        # self.x_samples = torch.unsqueeze(torch.tensor(x_samples), dim=1)
        # self.x_samples = self.x_samples.type(torch.float)
        # self.y_samples = (torch.tensor(y_samples)).type(torch.float)
        self.x_samples = Variable(torch.Tensor(x_samples).type(torch.float))
        self.y_samples = Variable(torch.Tensor(y_samples).type(torch.float))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x_samples)

    def __getitem__(self, idx):
        x = self.x_samples[idx]
        y = self.y_samples[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


if __name__ == "__main__":
    x = [[1.0], [2.0], [3.0]]
    y = [[1.0], [4.0], [9.0]]

    x = [[i] for i in range(10)]
    y = [[i ** 2] for i in range(10)]
    dataset = RegressionDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    x, y = next(iter(dataloader))
    print(x, y)
    # print(len(dataset))
    # print(dataset[1:10])
