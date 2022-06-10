from interpolation.regression_nn import RegressionNN
from random import uniform
import numpy as np
from interpolation.interpolate_system import odeint_interpolate
import matplotlib.pyplot as plt

INTERVAL_LOW = 0
INTERVAL_HIGH = 1


def system(y, t):
    S, I, R = y
    dSdt = -0.2 * S * I / 1000
    dIdt = 0.2 * S * I / 1000 - 0.1 * I
    dRdt = 0.1 * I
    return dSdt, dIdt, dRdt


def check_variable_exists(m, var_idx, y, a, b, threshold=1e-6):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # * Checks if variable with index var_idx is present in f(t,y1,y2,y3,...,yn)
    #   in interval [a,b]
    # * Main Idea:
    #   - Set X = (t0,y0) where t0 in [a,b] and y0 = (y1(t0),...,yn(t0))
    #   - Compute model(X)
    #   - Change only the variable with index var_idx
    #   - Compute again model(X)
    #   - Repeat iterations
    #   - Compute the variance of the results: If the variance is close to 0 then
    #       is possible that the variable with index var_idx is not present in f(t,y)
    # ---------------------------------------------------
    t0 = uniform(a, b)
    Y = [float(yi(t0)) for yi in y]
    X = [t0] + Y
    init_value = m.eval_model(X)[0]
    model_values = [init_value]

    iterations = 1000
    for _ in range(iterations):
        t = uniform(a, b)
        if var_idx == 0:
            X[0] = t
        else:
            X[var_idx] = float(y[var_idx - 1](t))
        new_value = m.eval_model(X)[0]

        model_values.append(new_value)

    variance = np.var(model_values)
    return variance
    # if variance < threshold:
    #     return False
    # return True


def check_variable_exists_epsilon(m, var_idx, y, a, b, epsilon=1, samples=1000):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # * Checks if variable with index var_idx is present in f(t,y1,y2,y3,...,yn)
    #   in interval [a,b]
    # * Main Idea:
    #   - Same as above, but the changes in the variable with index var_idx
    #       is in a epsilon neigh...
    # ---------------------------------------------------
    t0 = uniform(a, b)
    Y = [float(yi(t0)) for yi in y]
    X = [t0] + Y
    if var_idx > 0:
        var_idx_values = np.linspace(
            Y[var_idx - 1] - epsilon, Y[var_idx - 1] + epsilon, samples
        )
    else:
        var_idx_values = np.linspace(t0 - epsilon, t0 + epsilon, samples)

    values = []
    for v in var_idx_values:
        X[var_idx] = float(v)
        new_value = m.eval_model(X)[0]
        values.append(new_value)

    variance = np.var(values)
    return variance


def check_var_exist_grad(m, var_idx, y, a, b, samples=1000):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # * Checks if variable with index var_idx is present in f(t,y1,y2,y3,...,yn)
    #   in interval [a,b]
    # * Main Idea:
    #   - Derivate the model with respect to the variable with index var_idx
    #   - If in the interval [a,b] the partial derivate is close to 0 then
    #   - If in the interval [a,b] the partial derivate is close to 0 then
    #       is possible that the variable with index var_idx is not present in f(t,y)
    #
    # ---------------------------------------------------
    values = []
    t = np.linspace(a, b, samples)
    for ti in t:
        Y = [float(yi(ti)) for yi in y]
        X = [float(ti)] + Y
        v = m.m_grad(var_idx, X)  # dN/d_idx(X)
        values.append(v)

    return np.mean(values)


def dependency_graph_epsilon(m, var_idx, y, a, b, samples=1000, label="", color="b"):
    t0 = uniform(a, b)
    Y = [float(yi(t0)) for yi in y]
    X = [t0] + Y
    X_norm = [t0] + [yi / 1000 for yi in Y]
    m1 = ExactModel()
    epsilon = 1
    var_idx_values = np.linspace(
        Y[var_idx - 1] - epsilon, Y[var_idx - 1] + epsilon, samples
    )
    exact_values = []
    result_values = []
    t = np.linspace(a, b, samples)
    for ti in t:
        # for value in var_idx_values:
        #     if var_idx == 0:
        #         X[0] = value
        #     else:
        #         X[var_idx] = float(value)
        #         X_norm[var_idx] = float(value) / 1000
        ti = float(ti)
        Y = [float(yi(ti)) for yi in y]
        X = [ti] + Y
        X_norm = [ti] + [yi / 1000 for yi in Y]

        v = m.eval_model(X_norm)[0]
        exact_values.append(m1.eval_model(X)[0])
        result_values.append(v)

    # print(exact_values)
    # print(result_values)
    plt.plot(
        # var_idx_values,
        t,
        result_values,
        color,
        label=label,
    )
    plt.plot(
        # var_idx_values,
        t,
        exact_values,
        "r",
        label="Exact(S) for same fixed variables",
    )

    plt.legend(loc="best")
    plt.xlabel(f"y_{var_idx}")
    plt.ylabel(f"N(y_{var_idx})")
    plt.grid()


def dependency_graph(m, var_idx, y, a, b, samples=1000, label="", color="b"):
    Y = []
    for yi in y:
        t0 = uniform(a, b)
        Y.append(float(yi(t0)))
    t0 = uniform(a, b)
    X = [t0] + Y

    t = np.linspace(a, b, samples)
    var_idx_values = t
    if var_idx > 0:
        var_idx_values = [float(y[var_idx - 1](ti)) for ti in t]
        var_idx_values.sort()

    result_values = []
    exact_values = []
    m1 = ExactModel()
    for value in var_idx_values:
        if var_idx == 0:
            X[0] = value
        else:
            X[var_idx] = value

        v = m.eval_model(X)[0]
        exact_values.append(m1.eval_model(X)[0])
        result_values.append(v)

    plt.plot(
        var_idx_values,
        result_values,
        color,
        label=label,
    )
    plt.plot(
        var_idx_values,
        exact_values,
        "r",
        label="Exact(S) for same fixed variables",
    )

    plt.legend(loc="best")
    plt.xlabel(f"y_{var_idx}")
    plt.ylabel(f"N(y_{var_idx})")
    plt.grid()


class ExactModel:
    def eval_model(self, X):
        return [-0.2 * X[1] * X[2] / 1000]


def get_time_norm_function(function, max_time):
    def function_norm_time(t):
        return function(t * max_time)

    return function_norm_time


def main():
    # m = RegressionNN.nn_regression(0, 10, [y], y_prime, int(1e5), 2048, 20)
    # print(
    #     check_variable_exists(
    #         m,
    #         0,
    #         [y],
    #         0,
    #         10,
    #     )
    # )

    # ------------------------------------------------
    #               Usign interpole system
    # ------------------------------------------------

    y_inter = odeint_interpolate(
        system, [1000, 1, 0], INTERVAL_LOW, INTERVAL_HIGH, 10000
    )
    y_inter_time_norm = [get_time_norm_function(yi, INTERVAL_HIGH) for yi in y_inter]

    def y_prime1(t):
        return float(-0.2 * y_inter_time_norm[0](t) * y_inter_time_norm[1](t) / 1000)

    m1 = RegressionNN.nn_regression(
        0, 1, y_inter_time_norm, y_prime1, int(1e5), 2048, 20
    )

    # print("----------------------------------------------------")
    # print(
    #     check_variable_exists(
    #         m1,
    #         2,
    #         y_inter,
    #         0,
    #         100,
    #     )
    # )
    # print(
    #     check_variable_exists_epsilon(
    #         m1,
    #         1,
    #         y_inter,
    #         0,
    #         100,
    #     )
    # )

    # print(
    #     check_var_exist_grad(
    #         m1,
    #         2,
    #         y_inter,
    #         INTERVAL_LOW,
    #         INTERVAL_HIGH,
    #     )
    # )

    # dependency_graph(
    #     m1,
    #     1,
    #     y_inter,
    #     INTERVAL_LOW,
    #     INTERVAL_HIGH,
    #     samples=1000,
    #     label="N(S) for t,I,R fixed",
    # )
    # dependency_graph(m1, 1, y_inter, 0, 100, samples=1000, label="N(S) for t,I,R fixed")
    # dependency_graph(m1, 1, y_inter, 0, 100, samples=1000, label="N(S) for t,I,R fixed")
    # dependency_graph(m1, 1, y_inter, 0, 100, samples=1000, label="N(S) for t,I,R fixed")
    # dependency_graph(m1, 1, y_inter, 0, 100, samples=1000, label="N(S) for t,I,R fixed")
    # dependency_graph(m1, 1, y_inter, 0, 100, samples=1000, label="N(S) for t,I,R fixed")

    dependency_graph_epsilon(
        m1,
        1,
        y_inter_time_norm,
        0,
        1,
        samples=1000,
        label="N(S) for t,I,R fixed",
    )
    plt.show()


if __name__ == "__main__":
    main()
