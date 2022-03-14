from regression_nn import nn_regression, eval_model, check_error
import matplotlib.pyplot as plt
from random import uniform


def lv1(X, alpha, beta):
    x, y = X
    return x * (alpha - beta * y)


def lv2(X, gamma, delta):
    x, y = X
    return -y * (gamma - delta * x)


def main():
    # x = [[i] for i in range(20, 30)]
    # y = [[i ** 2] for i in range(20, 30)]

    alpha = 0.04
    beta = 0.0005
    x1 = [uniform(10, 200) for _ in range(100)]
    x2 = [uniform(10, 200) for _ in range(100)]
    X = [[x, y] for (x, y) in zip(x1, x2)]
    y = [[lv2(x, alpha, beta)] for x in X]

    m = nn_regression(X, y, 5)
    # plt.plot(y, label="original")
    # plt.plot(eval_model(m, X), label="predicted")
    # plt.legend()
    # plt.show()

    # print(check_error(m, X, y))

    x1 = [uniform(10, 200) for _ in range(1000)]
    x2 = [uniform(10, 200) for _ in range(1000)]
    X = [[x, y] for (x, y) in zip(x1, x2)]
    y = [[lv2(x, alpha, beta)] for x in X]
    plt.plot(y, label="original")
    plt.plot(eval_model(m, X), label="predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
