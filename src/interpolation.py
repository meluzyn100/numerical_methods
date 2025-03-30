import numpy as np


def lagrange_interpolation(x, xs, ys):
    """
    Compute the Lagrange interpolation polynomial at given points x.

    Parameters:
        x (float or array-like): The points at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float or array-like: The interpolated values at x.
    """
    x = np.asarray(x)
    n = len(xs)
    interpolated_values = np.sum(
        [ys[i] * lagrange_basis(x, xs, i, n) for i in range(n)], axis=0
    )
    return interpolated_values


def lagrange_basis(x, xs, i, n):
    """
    Compute the i-th Lagrange basis polynomial at given points x.

    Parameters:
        x (float or array-like): The points at which to evaluate the basis polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        i (int): The index of the basis polynomial.
        n (int): The total number of data points.

    Returns:
        float or array-like: The values of the i-th basis polynomial at x.
    """
    x = np.asarray(x)
    basis = [(x - xs[j]) / (xs[i] - xs[j]) for j in range(n) if i != j]
    return np.prod(basis, axis=0)


def vandermonde_interpolation(x, xs, ys):
    """
    Compute the Vandermonde interpolation polynomial at given points x.

    Parameters:
        x (float or array-like): The points at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float or array-like: The interpolated values at x.
    """
    x = np.asarray(x)
    n = len(xs)
    V = [[xs[j] ** i for i in range(n)] for j in range(n)]

    coeffs = np.linalg.solve(V, ys)

    interpolated_values = np.sum([coeffs[i] * (x**i) for i in range(n)], axis=0)
    return interpolated_values


def newton_interpolation(x, xs, ys):
    """
    Compute the Newton interpolation polynomial at given points x.

    Parameters:
        x (float or array-like): The points at which to evaluate the polynomial.
        xs (list or array-like): The x-coordinates of the data points.
        ys (list or array-like): The y-coordinates of the data points.

    Returns:
        float or array-like: The interpolated values at x.
    """
    x = np.asarray(x)
    n = len(xs)
    divided_differences = compute_divided_differences(xs, ys)

    interpolated_values = divided_differences[0] * np.ones_like(x)
    product_term = np.ones_like(x)
    for i in range(1, n):
        product_term *= x - xs[i - 1]
        interpolated_values += divided_differences[i] * product_term

    return interpolated_values


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


def chebyshev_interpolation(x, func, interval, num_points, method):
    """
    Interpolate a function using Chebyshev nodes.

    Parameters:
        x (float or array-like): The points at which to evaluate the polynomial.
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

    return method(x, chebyshev_nodes, ys)


def cubic_spline_segment(x, xk, a, b, c, d):
    """
    Evaluate a cubic spline segment at a given point.

    Parameters:
        x (float): The point at which to evaluate the spline segment.
        xk (float): The knot corresponding to the segment.
        a, b, c, d (float): Coefficients of the cubic spline segment.

    Returns:
        float: The value of the spline segment at x.
    """
    return a + b * (x - xk) + c * (x - xk) ** 2 + d * (x - xk) ** 3


def cubic_spline(x, knots, a, b, c, d):
    """
    Evaluate the cubic spline at given points.

    Parameters:
        x (float or array-like): The points at which to evaluate the spline.
        knots (list or array-like): The x-coordinates of the spline knots.
        a, b, c, d (array-like): Coefficients of the cubic spline segments.

    Returns:
        array-like: The values of the spline at x.
    """
    x = np.atleast_1d(x)
    result = []
    for xi in x:
        for i in range(len(knots) - 1):
            if knots[i] <= xi < knots[i + 1]:
                result.append(
                    cubic_spline_segment(xi, knots[i], a[i], b[i], c[i], d[i])
                )
                break
        else:
            result.append(
                cubic_spline_segment(xi, knots[-1], a[-1], b[-1], c[-1], d[-1])
            )
    return np.array(result)


def compute_cubic_spline_coefficients(knots, values):
    """
    Compute the coefficients of a cubic spline.

    Parameters:
        knots (list or array-like): The x-coordinates of the spline knots.
        values (list or array-like): The y-coordinates of the spline knots.

    Returns:
        tuple: Coefficients (a, b, c, d) of the cubic spline.
    """
    n = len(values)
    h = np.diff(knots)
    a = np.array(values)

    # Construct the system of equations for c coefficients
    rhs = [
        3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])
        for i in range(1, n - 1)
    ]
    rhs = [0] + rhs + [0]
    diag = [1] + [2 * (h[i] + h[i + 1]) for i in range(n - 2)] + [1]
    diag_upper = [0] + h[1:-1].tolist()
    diag_lower = h[:-1].tolist() + [0]

    A = np.zeros((n, n))
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], diag_upper)
    np.fill_diagonal(A[:, 1:], diag_lower)

    c = np.linalg.solve(A, rhs)

    # Compute b and d coefficients
    b = [
        (a[i + 1] - a[i]) / h[i] - h[i] / 3 * (2 * c[i] + c[i + 1])
        for i in range(n - 1)
    ]
    d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n - 1)]

    return a, np.array(b), np.array(c), np.array(d)


def cubic_spline_function(knots, values):
    """
    Create a cubic spline interpolation function.

    Parameters:
        knots (list or array-like): The x-coordinates of the spline knots.
        values (list or array-like): The y-coordinates of the spline knots.

    Returns:
        callable: A function representing the cubic spline interpolation.
    """
    a, b, c, d = compute_cubic_spline_coefficients(knots, values)
    return lambda x: cubic_spline(x, knots, a, b, c, d)
