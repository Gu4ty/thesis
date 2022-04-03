from regression_nn import nn_regression_uns, eval_model, check_error
from random import uniform
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint


def y(t):
    return math.exp(-(t ** 2))


def y_prime(t):
    return -2 * t * y(t)


def system(t, y, m):
    s = eval_model(m, [[t, y[0]]])
    return s[0][0]


def main():
    m = nn_regression_uns(0, 10, [y], y_prime, 1000)
    t = np.linspace(0, 10, 1000)
    # y_pt = [y_prime(ti) for ti in t]
    y_t = [y(ti) for ti in t]
    # X = [[x, y] for (x, y) in zip(t, y_t)]
    # plt.plot(t, y_pt, label="original")
    # plt.plot(t, eval_model(m, X), label="predicted")
    # print("Y: ", y_p)
    # print("MODEL: ", eval_model(m, X))
    y0 = 1
    sol = odeint(system, y0, t, args=(m,), tfirst=True)
    plt.plot(t, y_t, label="original")
    plt.plot(t, sol, label="odeint")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
