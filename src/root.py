import numpy as np


def bisection_method(f, a, b, tolerance=1e-13):
    """
    Perform the bisection method to find a root of the function f in the interval [a, b].

    Parameters:
        f (callable): The function for which the root is to be found.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tolerance (float): The tolerance for the root approximation. Default is 1e-13.

    Returns:
        tuple: A tuple containing the root approximation and a list of midpoints during the iterations.

    Raises:
        ValueError: If the function does not satisfy the condition f(a) * f(b) < 0.
    """
    if f(a) * f(b) >= 0:
        raise ValueError(
            "The function must have opposite signs at the endpoints of the interval [a, b]."
        )

    midpoints = []
    while True:
        c = (a + b) / 2
        midpoints.append(c)

        if abs(f(c)) < tolerance:
            return c, midpoints

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c


def compute_log_differences(midpoints, root):
    """
    Compute the log differences of midpoint approximations for convergence analysis.

    Parameters:
        midpoints (list): A list of midpoint approximations from the bisection method.
        root (float): The actual root of the function.

    Returns:
        tuple: Two lists containing the log differences for current and next iterations.
    """
    log_differences_current = []
    log_differences_next = []

    for i in range(len(midpoints) - 1):
        current_diff = abs(midpoints[i] - root)
        next_diff = abs(midpoints[i + 1] - root)

        if current_diff == 0 or next_diff == 0:
            continue

        log_differences_current.append(np.log(current_diff))
        log_differences_next.append(np.log(next_diff))

    return log_differences_current, log_differences_next


def convergence_rate(midpoints, root):
    """
    Calculate the convergence rate of the bisection method.

    Parameters:
        midpoints (list): A list of midpoint approximations from the bisection method.
        root (float): The actual root of the function.

    Returns:
        tuple: A tuple containing the slope (rate of convergence) and the intercept of the fitted line.
    """
    log_differences_current, log_differences_next = compute_log_differences(
        midpoints, root
    )
    return np.polyfit(log_differences_current, log_differences_next, 1)


def transform_midpoints(midpoints, root):
    """
    Transform the midpoint approximations for convergence analysis.

    Parameters:
        midpoints (list): A list of midpoint approximations from the bisection method.
        root (float): The actual root of the function.

    Returns:
        tuple: Two lists containing the transformed log differences for current and next iterations.
    """
    return compute_log_differences(midpoints, root)


def derivative(f, x, dx=1e-4):
    """
    Compute the numerical derivative of a function at a given point.

    Parameters:
        f (callable): The function for which the derivative is to be computed.
        x (float): The point at which the derivative is evaluated.
        dx (float): A small increment for numerical differentiation. Default is 1e-4.

    Returns:
        float: The numerical derivative of the function at the given point.
    """
    return (f(x + dx) - f(x)) / dx


def newton_method(f, x0, tolerance=1e-9):
    """
    Perform Newton's method to find a root of the function f.

    Parameters:
        f (callable): The function for which the root is to be found.
        x0 (float): The initial guess for the root.
        tolerance (float): The tolerance for the root approximation. Default is 1e-9.

    Returns:
        list: A list of approximations for the root during the iterations.
    """
    approximations = [x0]
    while True:
        current = approximations[-1]
        next_approx = current - f(current) / derivative(f, current)
        approximations.append(next_approx)

        if abs(f(next_approx)) < tolerance:
            break

    return approximations


def secant_method(f, x0, x1, max_iterations=10, tolerance=1e-11):
    """
    Perform the secant method to find a root of the function f.

    Parameters:
        f (callable): The function for which the root is to be found.
        x0 (float): The first initial guess for the root.
        x1 (float): The second initial guess for the root.
        max_iterations (int): The maximum number of iterations to perform. Default is 10.
        tolerance (float): The tolerance for the root approximation. Default is 1e-9.

    Returns:
        tuple: A tuple containing the final root approximation and lists of x0, x1, and x2 values
               during the iterations for analysis.

    Raises:
        ValueError: If the function values at x0 and x1 are equal, causing division by zero.
    """
    x0_values = []
    x1_values = []
    x2_values = []

    for _ in range(max_iterations):
        if f(x1) == f(x0):
            raise ValueError(
                "Division by zero encountered in the secant method. Ensure f(x0) != f(x1)."
            )

        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0_values.append(x0)
        x1_values.append(x1)
        x2_values.append(x2)

        if abs(x2 - x1) < tolerance:
            break

        x0, x1 = x1, x2

    return x0_values
