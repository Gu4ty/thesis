import torch
from torch import nn
from random import uniform
from torch.autograd import Variable
from interpolation.regression_dataset import RegressionDataset
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------
# |             Neural Network class to interpolate F(t,y)
# |     ** Methods overview:
# |         * nn_regression:
# |             - static method
# |             - generates an instance of the class m(model) such that m(t,y) ~ F(t,y)
# |             - m(t,y) ~ F(t,y) in the interval [a,b]
# |         * eval_model:
# |             - Given X = (t, y)
# |             - Evaluate m(X) and returns a numpy array
# |         * m_grad:
# |             - Given X = (t, y) and an index i
# |             - Computes dm(p1,p2,...,pn)/dpi(X)
# ----------------------------------------------------------------------------


class RegressionNN(nn.Module):
    def __init__(self, x_dimension):
        super(RegressionNN, self).__init__()

        self.linear_LeakyReLU_stack0 = nn.Sequential(
            torch.nn.Linear(x_dimension, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 1),
        )

        self.linear_LeakyReLU_stack1 = nn.Sequential(
            torch.nn.Linear(x_dimension, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
        )

        self.linear_LeakyReLU_stack2 = nn.Sequential(
            torch.nn.Linear(x_dimension, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
        )
        self.LeakyReluFey = nn.Sequential(
            torch.nn.Linear(x_dimension, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        logits = self.LeakyReluFey(x)
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
        epochs=100,
        dataset_train=None,
        dataset_test=None,
        learning_rate=0.005,
    ):
        # Defining the model topology
        model = RegressionNN(len(y) + 1)  # N = f(t, y1(t),y2(t),y3(t),..., yn(y))
        # Choosing an optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.0
        )
        # And the loss function
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss

        # Generating the datasets for training and testing
        # -------------------------------------------------------------------------------
        if not dataset_train:
            dataset_train = RegressionDataset(
                y, y_prime, interval_low, interval_high, samples_number * 80 // 100
            )
        if not dataset_test:
            dataset_test = RegressionDataset(
                y,
                y_prime,
                interval_low,
                interval_high,
                samples_number * 20 // 100,
            )
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        # -------------------------------------------------------------------------------

        # Training epochs
        for e in range(epochs):
            print(f"epoch: {e}")
            model.train_loop(dataloader_train, loss_fn, optimizer)  # Adjust weights
            model.test_loop(dataloader_test, loss_fn)  # Compute average loss
        # -------------------------------------------------------------------------------

        return model

    def train_loop(self, dataloader, loss_fn, optimizer):
        for (X, y) in dataloader:
            pred = self(X)
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
        return test_loss

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
            model.test_loop(dataloader_test, loss_fn)

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
        y_t = [[float(function(ti)) for function in y] for ti in t]
        y_prime_t = [[y_prime(ti)] for ti in t]
        y_prime_t = Variable(torch.Tensor(y_prime_t).type(torch.float))
        # arguments = t + y_t
        arguments = [[ti] + y_ti for (ti, y_ti) in zip(t, y_t)]
        arguments = Variable(torch.Tensor(arguments).type(torch.float))

        pred = self(arguments)
        loss = loss_fn(pred, y_prime_t)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval_model(self, x):
        x = torch.tensor(x)
        return self(x).detach().numpy()

    def m_grad(self, i, x):
        # Computes dN(p1,p2,...,pn)/dpi(X)
        r = torch.tensor(x, requires_grad=True)
        y = self(r)
        y.backward(retain_graph=True)
        g = r.grad  # dy/p0(x) dy/p1(x) dy/p2(x)....dy/pn(x)
        return float(g[i])

    def deriv(self, i, X, h=0.001):
        y1 = self.eval_model(X)[0]
        X[i] = X[i] + h
        y2 = self.eval_model(X)[0]
        return (y2 - y1) / h
