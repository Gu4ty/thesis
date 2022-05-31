from regression_nn import RegressionNN
import math
from random import uniform
import numpy as np


# def y(t):
#     return 10 * t


# def y_prime(t):
#     return 10


def y(t):
    return math.exp(t)


def y_prime(t):
    return math.exp(t)


def f(t, y):
    return y


def check_variable_exists(m, var_idx, y, a, b, threshold=1e-6):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # Checks if variable with index var_idx is present in f(t,y1,y2,y3,...,yn) in interval [a,b]
    # ---------------------------------------------------
    t0 = uniform(a, b)
    Y = [yi(t0) for yi in y]
    X = [t0] + Y
    init_value = m.eval_model([X])[0][0]
    model_values = [init_value]
    print(f't  = {t0}, y = {X[-1]} N(t,y) = {init_value}", " f(t,y) = {f(t0,X[-1])}')
    iterations = 1000
    for _ in range(iterations):
        t = uniform(a, b)
        if var_idx == 0:
            X[0] = t
        else:
            X[var_idx] = y[var_idx - 1](t)
        new_value = m.eval_model([X])[0][0]
        print(f"X: {X}")
        print(f't  = {t}, y = {X[-1]} N(t,y) = {new_value}", " f(t,y) = {f(t,X[-1])}')
        model_values.append(new_value)

    variance = np.var(model_values)
    return variance
    if variance < threshold:
        return False
    return True


def main():
    # print(test_check(f, y, 1, 10))
    m = RegressionNN.nn_regression(1, 10, [y], y_prime, int(1e5), 2048, 20)
    # m = RegressionNN.nn_regression_uns(0, 1, [y], y_prime, 1000, 5000)
    # print(m)
    print(
        check_variable_exists(
            m,
            0,
            [y],
            1,
            10,
        )
    )


if __name__ == "__main__":
    main()
