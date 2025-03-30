import numpy as np


def lagrange_interpolation(x, xs, ys):
    """
    Compute the Lagrange interpolation polynomial at a given point x.

    Parameters:
        x (float): The point at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float: The interpolated value at x.
    """
    n = len(xs)
    interpolated_value = sum(ys[i] * lagrange_basis(x, xs, i, n) for i in range(n))
    return interpolated_value


def lagrange_basis(x, xs, i, n):
    """
    Compute the i-th Lagrange basis polynomial at a given point x.

    Parameters:
        x (float): The point at which to evaluate the basis polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        i (int): The index of the basis polynomial.
        n (int): The total number of data points.

    Returns:
        float: The value of the i-th basis polynomial at x.
    """
    basis = [(x - xs[j]) / (xs[i] - xs[j]) for j in range(n) if i != j]
    return np.prod(basis)


def vandermonde_interpolation(x, xs, ys):
    """
    Compute the Vandermonde interpolation polynomial at a given point x.

    Parameters:
        x (float): The point at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float: The interpolated value at x.
    """
    n = len(xs)
    V = [[xs[j] ** i for i in range(n)] for j in range(n)]

    coeffs = np.linalg.solve(V, ys)

    interpolated_value = sum(coeffs[i] * (x**i) for i in range(n))
    return interpolated_value


def newton_interpolation(x, xs, ys):
    """
    Compute the Newton interpolation polynomial at a given point x.

    Parameters:
        x (float): The point at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float: The interpolated value at x.
    """
    n = len(xs)
    divided_differences = compute_divided_differences(xs, ys)

    interpolated_value = divided_differences[0]
    product_term = 1
    for i in range(1, n):
        product_term *= x - xs[i - 1]
        interpolated_value += divided_differences[i] * product_term

    return interpolated_value


def compute_divided_differences(xs, ys):
    """
    Compute the divided differences table for Newton interpolation.

    Parameters:
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        list: The divided differences coefficients.
    """
    n = len(xs)
    divided_differences = ys.copy()

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            divided_differences[j] = (
                divided_differences[j] - divided_differences[j - 1]
            ) / (xs[j] - xs[j - i])

    return divided_differences


def chebyshev_interpolation(func, interval, num_points, method):
    """
    Interpolate a function using Chebyshev nodes.

    Parameters:
        func (callable): The function to interpolate.
        interval (tuple): The interval (a, b) over which to interpolate.
        num_points (int): The number of Chebyshev nodes to use.
        method (callable): The interpolation method to use (e.g., lagrange_interpolation).

    Returns:
        callable: A function representing the interpolated polynomial.
    """
    a, b = interval
    chebyshev_nodes = [
        0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * num_points))
        for k in range(num_points)
    ]
    ys = [func(x) for x in chebyshev_nodes]

    def interpolated_polynomial(x):
        return method(x, chebyshev_nodes, ys)

    return interpolated_polynomial
