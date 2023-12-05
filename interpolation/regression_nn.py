import torch
from torch import nn
from random import uniform
from torch.autograd import Variable
from interpolation.regression_dataset import RegressionDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
        self.learn_history = {}
        self.learn_history["train_loss"] = []
        self.learn_history["test_loss"] = []

        self.SimpleFeySoft = nn.Sequential(
            torch.nn.Linear(x_dimension, 16),
            torch.nn.Softplus(),
            # torch.nn.Linear(128, 128),
            # torch.nn.Tanh(),
            torch.nn.Linear(16, 4),
            torch.nn.Softplus(),
            # torch.nn.Linear(64, 64),
            # torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
        )

        self.SimpleFeyRelu = nn.Sequential(
            torch.nn.Linear(x_dimension, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 1),
        )

        self.SoftPlusFey = nn.Sequential(
            torch.nn.Linear(x_dimension, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 1),
        )

        self.LeakyReluFey0 = nn.Sequential(
            torch.nn.Linear(x_dimension, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 1),
        )

        self.OneHiddenSoft = nn.Sequential(
            torch.nn.Linear(x_dimension, 300),
            torch.nn.Softplus(),
            torch.nn.Linear(300, 1),
        )

    def forward(self, x):
        logits = self.SoftPlusFey(x)
        return logits

    @staticmethod
    def nn_regression(
        y,
        y_prime,
        samples_number=int(1e6),
        batch_size=1000,
        epochs=100,
        dataset_train=None,
        dataset_test=None,
        include_time=True,
        normalization_factor=1,
        learning_rate=0.005,
    ):
        # Defining the model topology
        if include_time:
            model = RegressionNN(
                len(y) + 1
            )  # model = f(t, y1(t),y2(t),y3(t),..., yn(y))
        else:
            model = RegressionNN(len(y))  # model = f( y1(t),y2(t),y3(t),..., yn(y))

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()

        model = model.double()
        # Choosing an optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # And the loss function
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss

        # Generating the datasets for training and testing
        # -------------------------------------------------------------------------------
        if not dataset_train:
            dataset_train = RegressionDataset(
                y, y_prime, normalization_factor, samples_number * 80 // 100
            )
        if not dataset_test:
            dataset_test = RegressionDataset(
                y,
                y_prime,
                normalization_factor,
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
            model.train()
            train_loss = model.train_loop(
                dataloader_train, loss_fn, optimizer
            )  # Adjust weights
            model.learn_history["train_loss"].append(train_loss)
            model.eval()
            test_loss = model.test_loop(
                dataloader_test, loss_fn
            )  # Compute average loss
            model.learn_history["test_loss"].append(test_loss)
        # -------------------------------------------------------------------------------

        return model

    @staticmethod
    def nn_dataset_regression(
        dataset_train,
        dataset_test,
        batch_size=1000,
        epochs=100,
        learning_rate=0.005,
    ):
        # Defining the model topology
        model = RegressionNN(len(dataset_train.X[0]))
        model = model.double()

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print(
                "Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB"
            )
            print(
                "Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB"
            )

        # Choosing an optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # Choosing loss function
        loss_fn = torch.nn.MSELoss()  # this is for regression mean squared loss

        # Generating the dataloaders for training and testing
        # -------------------------------------------------------------------------------
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        # -------------------------------------------------------------------------------
        # Training epochs
        for e in range(epochs):
            if e % 10 == 0:
                print(f"epoch: {e}")
            model.train()
            train_loss = model.train_loop(
                dataloader_train, loss_fn, optimizer, e % 10 == 0 or e == (epochs - 1)
            )  # Adjust weights
            model.learn_history["train_loss"].append(train_loss)
            model.eval()
            test_loss = model.test_loop(
                dataloader_test, loss_fn, e % 10 == 0 or e == (epochs - 1)
            )  # Compute average loss
            model.learn_history["test_loss"].append(test_loss)
        # -------------------------------------------------------------------------------
        return model

    def train_loop(self, dataloader, loss_fn, optimizer, print_loss=True):
        num_batches = len(dataloader)
        train_loss = 0

        for (X, y) in dataloader:
            pred = self(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"loss: {loss}")
        train_loss /= num_batches
        if print_loss:
            print(f"train_loss: {train_loss}")
        return train_loss

    def test_loop(self, dataloader, loss_fn, print_loss=True):
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        if print_loss:
            print(f"test_loss: {test_loss}")
        return test_loss

    def plot_learn_history(self, start=0, end=100):
        train_loss = self.learn_history["train_loss"]
        test_loss = self.learn_history["test_loss"]
        epochs = [i + 1 for i in range(len(train_loss))]
        plt.plot(epochs[start:end], train_loss[start:end], label="train_loss")
        plt.plot(epochs[start:end], test_loss[start:end], label="test_loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("training history")
        plt.grid()
        plt.legend(loc="best")
        plt.show()

    def eval_model(self, x):
        x = torch.tensor(x).to("cuda")
        # print(f" X = {x} ------ N(X) = {self(x).detach().numpy()}")
        return self(x).cpu().detach().numpy()

    def m_grad(self, i, x):
        # Computes dN(p1,p2,...,pn)/dpi(x)
        r = torch.tensor(x, requires_grad=True, device="cuda")
        y = self(r)
        y.backward(retain_graph=True)
        g = r.grad  # dy/dp0(x) dy/dp1(x) dy/dp2(x)....dy/dpn(x)
        return g[i].cpu().detach().numpy()

    def deriv(self, i, X, h=0.001):
        y1 = self.eval_model(X)[0]
        X[i] = X[i] + h
        y2 = self.eval_model(X)[0]
        return (y2 - y1) / h
