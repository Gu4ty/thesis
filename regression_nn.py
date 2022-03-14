import torch
import numpy as np
from regression_dataset import RegressionDataset
from torch import nn
from torch.utils.data import DataLoader
from random import uniform

# import matplotlib.pyplot as plt
from torch.autograd import Variable


class RegressionNN(nn.Module):
    def __init__(self, x_dimension):
        super(RegressionNN, self).__init__()
        self.linear_LeakyReLU_stack = nn.Sequential(
            torch.nn.Linear(x_dimension, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 1),
        )
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        logits = self.linear_LeakyReLU_stack(x)
        return logits
        # return self.linear(x)


def nn_regression(x, y, batch_size, epochs=500, learning_rate=0.01):
    dataset = RegressionDataset(x, y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RegressionNN(len(x[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss

    for _ in range(epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer)
        # test_loop(test_dataloader, model, loss_fn)

    return model


def nn_regression_uns(
    interval_low, interval_high, y, y_prime, batch_size, epochs=500, learning_rate=0.01
):
    model = RegressionNN(len(y) + 1)  # N = f(t, y1(t),y2(t),y3(t),..., yn(y))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss
    for _ in range(epochs):
        train_loop_uns(
            interval_low,
            interval_high,
            y,
            y_prime,
            batch_size,
            model,
            loss_fn,
            optimizer,
        )

    return model


def train_loop_uns(
    interval_low,
    interval_high,
    y,
    y_prime,
    batch_size,
    model: RegressionNN,
    loss_fn,
    optimizer,
):
    t = [uniform(interval_low, interval_high) for _ in range(batch_size)]
    y_t = [[function(ti) for function in y] for ti in t]
    y_prime_t = [[y_prime(ti)] for ti in t]
    y_prime_t = Variable(torch.Tensor(y_prime_t).type(torch.float))
    # arguments = t + y_t
    arguments = [[ti] + y_ti for (ti, y_ti) in zip(t, y_t)]
    arguments = Variable(torch.Tensor(arguments).type(torch.float))
    pred = model(arguments)
    # print("*********************")
    # print("X: ", arguments)
    # print("Y: ", y_prime_t)
    # print("PRED: ", pred)
    # print("*********************")
    loss = loss_fn(pred, y_prime_t)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_loop(dataloader, model, loss_fn, optimizer):
    print("************************")
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        print("X: ", X)
        print("Y: ", y)
        pred = model(X)
        print("PRED: ", pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):

    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            print(f"In test loop: y = {y}")
            print(f"In test loop: yPred = {pred}")
            test_loss += loss_fn(pred, y).item()

    print(f"Avg Test loss per batch = {test_loss/num_batches}")


def eval_model(model, x):
    x_samples = Variable(torch.Tensor(x).type(torch.float))
    return model(x_samples).detach().numpy()


def check_error(model, x, y):
    y_pred = eval_model(model, x)
    print(f"Y pred: {y_pred}")
    print("***************************************")
    print(f"Y: {y}")
    return np.square(np.subtract(y, y_pred)).mean()


if __name__ == "__main__":
    x = [[i] for i in range(1, 1000)]
    y = [[i ** 2] for i in range(1, 1000)]
    # x = [[i, i] for i in range(1, 50)]
    # y = [[i[0] + i[1]] for i in x]
    m = nn_regression(x, y, 50)

    # plt.plot(x, y, label="original")
    # plt.plot(x, eval_model(m, x), label="nn")
    # plt.legend()
    # plt.show()
