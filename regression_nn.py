import torch
from torch import nn
from random import uniform
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

    @staticmethod
    def nn_regression_uns(
        interval_low,
        interval_high,
        y,
        y_prime,
        batch_size,
        epochs=500,
        learning_rate=0.01,
    ):
        model = RegressionNN(len(y) + 1)  # N = f(t, y1(t),y2(t),y3(t),..., yn(y))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss
        for _ in range(epochs):
            model.train_loop_uns(
                interval_low,
                interval_high,
                y,
                y_prime,
                batch_size,
                loss_fn,
                optimizer,
            )

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
        pred = self(arguments)
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

    def eval_model(self, x):
        x_samples = Variable(torch.Tensor(x).type(torch.float))
        return self(x_samples).detach().numpy()
