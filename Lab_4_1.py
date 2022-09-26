import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from scipy.optimize import minimize, least_squares, differential_evolution


def data_func(x: float) -> float:
    """
    Function to return result of counting data function

    :param x: param x
    :type x: float
    :return: result of counting
    :rtype: float
    """

    return 1 / (x ** 2 - 3 * x + 2)


def rational_func(x: list, a: float, b: float, c: float, d: float) -> np.array:
    """
    Function to return result of counting data function

    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :param c: coefficient c
    :param d: coefficient d
    :type x: list
    :type a: float
    :type b: float
    :type c: float
    :type d: float
    :return: result of counting
    :rtype: np.array
    """

    return (a * np.array(x) + b) / (np.array(x) ** 2 + c * np.array(x) + d)


def nelder_mead(func: Callable, eps: float = 0.001) -> tuple:
    """
    Function of Nelder Mead's method to find minimum of input function

    :param func: input function
    :param eps: epsilon
    :type func: Callable
    :type eps: float
    :return: best params a and b to minimize function
    :rtype: tuple
    """

    best = minimize(func, x0=[0, 0, 0, 0], method='Nelder-Mead', tol=eps)
    return best['x'][0], best['x'][1], best['x'][2], best['x'][3]


def levenberg_marquardt(func: Callable) -> tuple:
    """
    Function of levenberg-marquardt's method to find minimum of input function

    :param func: input function
    :type: func: Callable
    :return: the best a and b coefficients
    :rtype: tuple
    """

    res = least_squares(func, np.array([1, 1, 1, 1]), method='lm')

    return res['x'][0], res['x'][1], res['x'][2], res['x'][3]


def diff_evolution(func: Callable) -> tuple:
    """
    Function of differential evolution method to find minimum of input function

    :param func: input function
    :type: func: Callable
    :return: the best a and b coefficients
    :rtype: tuple
    """

    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    res = differential_evolution(func, bounds)

    return res['x'][0], res['x'][1], res['x'][2], res['x'][3]


x_list = [3 * k / 1000 for k in range(1001)]
y_list = [(-100 + random.normalvariate(0, 1)) if data_func(elem) < -100 else
          (data_func(elem) + random.normalvariate(0, 1)) if abs(data_func(elem) <= 100) else
          (100 + random.normalvariate(0, 1)) for elem in x_list]


def errors_func_rational(params: list) -> np.array:
    """
    Function for finding errors function of linear function for Nelder Mead's method
    (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: errors sum
    :rtype: np.array
    """

    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return np.sum((((a * np.array(x_list) + b) / (np.array(x_list) ** 2 + c * np.array(x_list) + d)) - np.array(y_list)) ** 2)


def errors_func_ration_lev_marq(params: list) -> list:
    """
    Function for finding errors function of rational function for levenberg marquardt's method
    (because of specific of scipy)

    :param params: params a, b, c, d
    :type params: list
    :return: [errors sum, errors sum, errors sum, errors sum]
    :rtype: list
    """

    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return [np.sum((((a * np.array(x_list) + b) / (np.array(x_list) ** 2 + c * np.array(x_list) + d)) - np.array(y_list)) ** 2),
            np.sum((((a * np.array(x_list) + b) / (np.array(x_list) ** 2 + c * np.array(x_list) + d)) - np.array(y_list)) ** 2),
            np.sum((((a * np.array(x_list) + b) / (np.array(x_list) ** 2 + c * np.array(x_list) + d)) - np.array(y_list)) ** 2),
            np.sum((((a * np.array(x_list) + b) / (np.array(x_list) ** 2 + c * np.array(x_list) + d)) - np.array(y_list)) ** 2)]


# Plot of results
plt.scatter(x_list, y_list, color='orange', s=10)
plt.plot(x_list, rational_func(x_list, *nelder_mead(errors_func_rational)), label="nelder mead")
plt.plot(x_list, rational_func(x_list, *levenberg_marquardt(errors_func_ration_lev_marq)), label="levenberg marquardt")
plt.plot(x_list, rational_func(x_list, *diff_evolution(errors_func_rational)), label="differential evolution")
plt.legend()
plt.show()

