import torch
from torch import nn
from random import uniform
from torch.autograd import Variable
from regression_dataset import RegressionDataset
from torch.utils.data import DataLoader


class RegressionNN(nn.Module):
    # def __init__(self, x_dimension):
    #     super(RegressionNN, self).__init__()
    #     self.linear_LeakyReLU_stack = nn.Sequential(
    #         torch.nn.Linear(x_dimension, 300),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Linear(300, 300),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Linear(300, 1),
    #     )
    #     self.linear = torch.nn.Linear(1, 1)

    def __init__(self, x_dimension):
        super(RegressionNN, self).__init__()
        self.linear_LeakyReLU_stack = nn.Sequential(
            torch.nn.Linear(x_dimension, 128),
            torch.nn.LeakyReLU(),
            # torch.nn.Softplus(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            # torch.nn.Softplus(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.Softplus(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.Softplus(),
            # torch.nn.Linear(256, 128),
            # torch.nn.Softplus(),
            # torch.nn.Linear(128, 128),
            # torch.nn.Softplus(),
            torch.nn.Linear(64, 1),
        )
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        logits = self.linear_LeakyReLU_stack(x)
        return logits
        # return self.linear(x)

    @staticmethod
    def nn_regression(
        interval_low,
        interval_high,
        y,
        y_prime,
        samples_number=int(1e6),
        batch_size=1000,
        epochs=5,
        learning_rate=0.005,
    ):
        model = RegressionNN(len(y) + 1)  # N = f(t, y1(t),y2(t),y3(t),..., yn(y))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss
        dataset_train = RegressionDataset(
            y, y_prime, interval_low, interval_high, samples_number * 80 // 100
        )
        dataset_test = RegressionDataset(
            y, y_prime, interval_low, interval_high, samples_number * 20 // 100
        )

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        for e in range(epochs):
            print(f"epoc: {e}")
            model.train_loop(dataloader_train, loss_fn, optimizer)
            model.test_loop(dataloader_test, loss_fn)

        return model

    def train_loop(self, dataloader, loss_fn, optimizer):
        for (X, y) in dataloader:
            pred = self(X)
            # print(f"pred: {pred}      y: {y}")
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"loss: {loss}")

    def test_loop(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"Avg loss: {test_loss}")

    @staticmethod
    def nn_regression_uns(
        interval_low,
        interval_high,
        y,
        y_prime,
        batch_size,
        epochs=500,
        learning_rate=0.005,
    ):
        model = RegressionNN(len(y) + 1)  # N = f(t, y1(t),y2(t),y3(t),..., yn(y))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss
        dataset_test = RegressionDataset(
            y, y_prime, interval_low, interval_high, batch_size * epochs * 20 // 100
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        for e in range(epochs):
            print(f"epoch {e}")
            model.train_loop_uns(
                interval_low,
                interval_high,
                y,
                y_prime,
                batch_size,
                loss_fn,
                optimizer,
            )
            # model.test_loop(dataloader_test, loss_fn)

        return model

    def train_loop_uns(
        self,
        interval_low,
        interval_high,
        y,
        y_prime,
        batch_size,
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
        # print(f"t : {t}")
        # print(f"y : {y_t}")
        # print(f"y' : {y_prime_t}")
        # print(f"X : {arguments}")

        pred = self(arguments)
        # print(f"pred : {pred}")
        # print("*********************")
        # print("X: ", arguments)
        # print("Y: ", y_prime_t)
        # print("PRED: ", pred)
        # print("*********************")
        loss = loss_fn(pred, y_prime_t)
        # print(f"Loss: {loss}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval_model(self, x):
        x_samples = Variable(torch.Tensor(x).type(torch.float))
        return self(x_samples).detach().numpy()
