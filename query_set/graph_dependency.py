import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from query_set.utils import generate_values


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
        Y.append(yi(t0))
    t0 = uniform(a, b)
    X = [t0] + Y

    t = np.linspace(a, b, samples)
    var_idx_values = t
    if var_idx > 0:
        var_idx_values = [y[var_idx - 1](ti) for ti in t]
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


def dependency_graph_epsilon_mean(
    m,
    var_idx,
    dataset,
    normalization_factor,
    epsilon,
    samples=1000,
    label="",
    color="b",
    xlabel=None,
    ylabel=None,
    exact_model_for_debug=None,
):  # TODO: missing time treatment
    X = [x for x in dataset.X_means]
    X_norm = [xi / normalization_factor for xi in X]
    lower_bound = max(0, X[var_idx] - epsilon)
    upper_bound = min(normalization_factor, X[var_idx] + epsilon)

    result_values = []
    exact_values = []
    var_idx_values = np.linspace(lower_bound, upper_bound, samples)
    for value in var_idx_values:
        X[var_idx] = value
        X_norm[var_idx] = (value) / normalization_factor
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
        exact_label = None
        if label:
            exact_label = "Exact " + label
        plt.plot(
            var_idx_values,
            exact_values,
            color="r",
            label=exact_label,
        )

    if label:
        plt.legend(loc="best")
    if xlabel == None:
        plt.xlabel("X[var_idx]")
    else:
        plt.xlabel(xlabel)

    if ylabel == None:
        plt.ylabel("N(X[var_idx])")
    else:
        plt.ylabel(ylabel)

    plt.grid()


def dependency_graph_epsilon(
    m,
    var_idx,
    y,
    a,
    b,
    normalization_factor,
    epsilon,
    samples=1000,
    include_time=True,
    label="",
    color="b",
    xlabel=None,
    ylabel=None,
    exact_model_for_debug=None,
    plotter=plt,
):
    # ---------------------------------------------------
    # model = dy/dt = f(t, y1, y2, y3,..., yn) or model = dy/dt = f(y1,y2,...,yn) if time not included
    # var_idx:      idx 0   1   2   3 ..... n
    # * Graph a curve (yi, model(t0,y10, y20, .... , yi,....,yn0)) for yi variable
    #   and all the others inputs fixed
    # * Main Idea:
    #   - Set t = t0 and y = (y1(t0), y2(t0),...,yn(t0))
    #   - Then, do the same process as above, but the changes in the variable with index
    #       var_idx is in a epsilon neigh..
    # ---------------------------------------------------
    t0 = uniform(a, b)
    Y = [yi(t0) for yi in y]
    if include_time:
        X = [t0] + Y
        X_norm = [t0] + [yi / normalization_factor for yi in Y]
    else:
        X = Y
        X_norm = [yi / normalization_factor for yi in Y]

    print(t0, X)

    lower_bound = max(0, Y[var_idx - 1] - epsilon)
    upper_bound = min(normalization_factor, Y[var_idx - 1] + epsilon)
    if var_idx == 0:
        lower_bound = max(0, t0 - epsilon)
        upper_bound = min(1, t0 + epsilon)
        if not include_time:
            lower_bound = max(0, Y[0] - epsilon)
            upper_bound = min(normalization_factor, Y[0] + epsilon)

    var_idx_values = np.linspace(lower_bound, upper_bound, samples)

    result_values = []
    exact_values = []
    # print(f"selected t0: {t0}")
    # if var_idx > 0:
    #     print(f"Y[{var_idx - 1}](t0) = {Y[var_idx-1]}")
    for value in var_idx_values:
        if var_idx == 0 and include_time:
            X[0] = value
            X_norm[0] = value
        else:
            X[var_idx] = value
            X_norm[var_idx] = (value) / normalization_factor

        if exact_model_for_debug:
            exact_values.append(exact_model_for_debug.eval_model(X_norm)[0])
        v = m.eval_model(X_norm)[0]
        result_values.append(v)

    plotter.plot(
        var_idx_values,
        result_values,
        color,
        # label=label,
        label=r"$NN_1(t_0, \hat{x}, y(t_0)$)",
    )
    if exact_model_for_debug:
        exact_label = None
        if label:
            exact_label = "Exact " + label
        plotter.plot(
            var_idx_values,
            exact_values,
            color="r",
            label=r"$f_1(t_0, \hat{x}, y(t_0))$",
        )

    if label:
        plotter.legend(loc="best")
    if type(plotter) == type(plt):
        if xlabel == None:
            plotter.xlabel("X[var_idx]")
        else:
            plotter.xlabel(xlabel)

        if ylabel == None:
            plotter.ylabel("N(X[var_idx])")
        else:
            plotter.ylabel(ylabel)

    plotter.grid()
