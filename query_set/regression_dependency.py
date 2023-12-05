import numpy as np
import math
from random import uniform
from scipy import stats
from query_set.utils import generate_values

def rmsValue(arr, n):
    square = 0
    mean = 0.0
    root = 0.0
     
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
     
    #Calculate Mean
    mean = (square / (float)(n))
     
    #Calculate Root
    root = math.sqrt(mean)
     
    return root

def check_linear_dependency(
    m,
    var_idx,
    y,
    a,
    b,
    normalization_factor,
    epsilon,
    samples=1000,
    include_time=True,
    number_of_tests=30,
):
    avg_intercept = 0
    avg_r_value_square = 0
    avg_slope = 0
    for _ in range(number_of_tests):
        var_idx_values, result_values = generate_values(
            m, var_idx, y, a, b, normalization_factor, epsilon, samples, include_time
        )

        regression_line = stats.linregress(var_idx_values, result_values)
        avg_intercept += regression_line.intercept
        avg_r_value_square += regression_line.rvalue**2
        avg_slope += regression_line.slope
    avg_intercept /= number_of_tests
    avg_r_value_square /= number_of_tests
    avg_slope /= number_of_tests
    return avg_slope, avg_intercept, avg_r_value_square


def check_polynomial_dependency(
    m,
    var_idx,
    y,
    a,
    b,
    degree,
    normalization_factor,
    epsilon,
    samples=1000,
    include_time=True,
    number_of_tests=30,
):
    # var_idx_values, result_values = generate_values(
    #     m, var_idx, y, a, b, normalization_factor, epsilon, samples, include_time
    # )
    # Polynomial Regression
    def polyfit(x, y, degree):
        results = {}

        coeffs = np.polyfit(x, y, degree)

        # Polynomial Coefficients
        results["polynomial"] = coeffs.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x)  # or [p(z) for z in x]
        ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
        ssreg = np.sum(
            (yhat - ybar) ** 2
        )  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        results["determination"] = ssreg / sstot
        return results

    regression_results = []
    pearsons = []
    for _ in range(number_of_tests):
        var_idx_values, result_values = generate_values(
            m, var_idx, y, a, b, normalization_factor, epsilon, samples, include_time
        )
        regression = polyfit(var_idx_values, result_values, degree)
        regression_results.append(regression["polynomial"])
        pearsons.append( regression["determination"])
    
    n = len(regression_results)
    results = {}
    results["polynomial"] = []
    for d in range(degree + 1):
        values = []
        for poly in regression_results:
            values.append(poly[d])
        rms = rmsValue(values,n )  
        results["polynomial"].append(rms);
    
    results["determination"] = rmsValue(pearsons, n)
    return results
