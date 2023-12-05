# from interpolation.regression_nn import RegressionNN
from random import uniform, randint
import numpy as np


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
    #       it is possible that the variable with index var_idx is not present in f(t,y)
    #
    # ---------------------------------------------------
    values = []
    t = np.linspace(a, b, samples)
    for ti in t:
        Y = [yi(ti) for yi in y]
        X = [ti] + Y
        v = m.m_grad(var_idx, np.array(X))  # dN/d_idx(X)
        values.append(abs(v))

    return values


def check_var_exist_grad_dataset(m, var_idx, dataset):
    values = []
    number_of_elements = min(100000, len(dataset.X))
    for x in dataset.X[:number_of_elements]:
        v = m.m_grad(var_idx, np.array(x.cpu()))  # dN/d_idx(X)
        values.append(abs(v))

    return values


def check_var_exist_grad_dataset_epsilon(
    m, var_idx, dataset, normalization_factor, epsilon=5
):
    values_mean = []
    dataset_len = len(dataset.X)
    for _ in range(30):
        values = []
        pick = randint(0, dataset_len - 1)
        X = dataset.X[pick]
        lower_bound = max(0, X[var_idx] - epsilon)
        upper_bound = min(normalization_factor, X[var_idx] + epsilon)
        var_idx_values = np.linspace(lower_bound, upper_bound, 1000)
        for x in var_idx_values:
            X = dataset.X[pick]
            X[var_idx] = x
            v = m.m_grad(var_idx, np.array(X.cpu()))  # dN/d_idx(X)
            values.append(abs(v))
        values_mean.append(sum(values) / len(values))

    return values_mean
