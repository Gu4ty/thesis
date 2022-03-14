from regression_nn import nn_regression_uns, eval_model, check_error
from random import uniform
import matplotlib.pyplot as plt
import math


def y(t):
    return math.exp(-(t ** 2))


def y_prime(t):
    return -2 * t * y(t)


def main():
    m = nn_regression_uns(1, 10, [y], y_prime, 1000)
    t = [x / 10.0 for x in range(10, 100, 5)]
    y_pt = [y_prime(ti) for ti in t]
    y_t = [y(ti) for ti in t]
    X = [[x, y] for (x, y) in zip(t, y_t)]
    plt.plot(t, y_pt, label="original")
    plt.plot(t, eval_model(m, X), label="predicted")
    # print("Y: ", y_p)
    # print("MODEL: ", eval_model(m, X))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
