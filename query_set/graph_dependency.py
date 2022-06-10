import numpy as np
import matplotlib.pyplot as plt
from random import uniform


class ExactModel:
    def eval_model(self, X):
        return [-0.2 * (X[1] * 1000) * X[2] * 1000 / 1000]


def dependency_graph(m, var_idx, y, a, b, samples=1000, label="", color="b"):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # * Graph a curve (yi, model(t0,y10, y20, .... , yi,....,yn0)) for yi variable
    #   and all the others inputs fixed
    # * Main Idea:
    #   - Set t = t0 in [a,b].
    #   - Set each yi = yi0 = yi(t) for a t in [a,b]
    #   - Iterate for each possible value of the variable with index var_idx
    #       in the interval [a,b]
    #   - Computes the model(X) for X = (t, y)
    #   - Generate the graph (yi, model(t0,y10, y20, .... , yi,....,yn0))
    # ---------------------------------------------------
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


def dependency_graph_epsilon(
    m, var_idx, y, a, b, samples=1000, label="", color="b", exact_model_for_debug=None
):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn)
    # var_idx:      idx 0   1   2   3 ..... n
    # * Graph a curve (yi, model(t0,y10, y20, .... , yi,....,yn0)) for yi variable
    #   and all the others inputs fixed
    # * Main Idea:
    #   - Set t = t0 and y = (y1(t0), y2(t0),...,yn(t0))
    #   - Then, do the same process as above, but the changes in the variable with index
    #       var_idx is in a epsilon neigh..
    # ---------------------------------------------------
    t0 = uniform(a, b)
    Y = [float(yi(t0)) for yi in y]
    X = [t0] + Y
    X_norm = [t0] + [yi / 1000 for yi in Y]
    epsilon = 100
    var_idx_values = np.linspace(
        Y[var_idx - 1] - epsilon, Y[var_idx - 1] + epsilon, samples
    )
    result_values = []
    exact_values = []
    print(f"selected t0: {t0}")
    print(f"S(t0) = {Y[0]}")
    for value in var_idx_values:
        if var_idx == 0:
            X[0] = value
        else:
            X[var_idx] = float(value)
            X_norm[var_idx] = float(value) / 1000

        if exact_model_for_debug:
            exact_values.append(exact_model_for_debug.eval_model(X_norm)[0])
        v = m.eval_model(X_norm)[0]
        result_values.append(v)

    plt.plot(
        var_idx_values,
        result_values,
        color,
        label=label,
    )
    if exact_model_for_debug:
        plt.plot(
            var_idx_values,
            exact_values,
            color="r",
            label="Exact" + label,
        )

    plt.legend(loc="best")
    plt.xlabel("S")
    plt.ylabel("N(S)")
    plt.grid()
