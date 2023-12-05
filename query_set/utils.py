import numpy as np
from random import uniform


def generate_values(
    m, var_idx, y, a, b, normalization_factor, epsilon, samples, include_time=True
):
    t0 = uniform(a, b)
    Y = [yi(t0) for yi in y]
    if include_time:
        X = [t0] + Y
        X_norm = [t0] + [yi / normalization_factor for yi in Y]
    else:
        X = Y
        X_norm = [yi / normalization_factor for yi in Y]

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
    for value in var_idx_values:
        if include_time and var_idx == 0:
            X[0] = value
            X_norm[0] = value
        else:
            X[var_idx] = value
            X_norm[var_idx] = (value) / normalization_factor

        v = m.eval_model(X_norm)[0]
        result_values.append(v)

    return var_idx_values, result_values
