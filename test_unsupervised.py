from regression_nn import RegressionNN
import matplotlib.pyplot as plt
import math


# def y(t):
#     return math.exp(-(t ** 2))


# def y_prime(t):
#     return -2 * t * y(t)


def y(t):
    return (3 * math.exp(2 * t) - 1) / (1 - math.exp(2 * t))


def y_prime(t):
    return y(t) ** 2 + 4 * y(t) + 3


def main():
    m = RegressionNN.nn_regression_uns(1, 10, [y], y_prime, 1000)
    t = [x / 10.0 for x in range(10, 100, 5)]
    y_pt = [y_prime(ti) for ti in t]
    y_t = [y(ti) for ti in t]
    X = [[x, y] for (x, y) in zip(t, y_t)]
    plt.plot(t, y_pt, label="original")
    plt.plot(t, m.eval_model(X), label="predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
