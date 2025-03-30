import numpy as np


def f(x):
    """Function to compute |x| * exp(x)."""
    return np.abs(x) * np.exp(x)


def find_n(integration_method, func, a, b, analytical_value, tolerance=1e-5, step=100):
    """
    Find the number of intervals (n) required for the integration method
    to achieve the desired accuracy.

    Parameters:
        integration_method (callable): Numerical integration method.
        func (callable): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        analytical_value (float): Analytical value of the integral.
        tolerance (float): Desired accuracy (default: 1e-5).
        step (int): Increment step for n (default: 100).

    Returns:
        tuple: (n, error) where n is the number of intervals and error is the final error.
    """
    n = 1
    while True:
        numerical_result, _ = integration_method(func, a, b, n)
        error = abs(numerical_result - analytical_value)
        if error <= tolerance:
            break
        n += step
    return n, error


def int_mid(func, a, b, n):
    """
    Midpoint rule for numerical integration.

    Parameters:
        func (callable): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of intervals.

    Returns:
        tuple: (integral, step_size) where integral is the computed integral
               and step_size is the size of each interval.
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    midpoints = (x[:-1] + x[1:]) / 2
    integral = np.sum(func(midpoints) * h)
    return integral, h


def int_trap(func, a, b, n):
    """
    Trapezoidal rule for numerical integration.

    Parameters:
        func (callable): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of intervals.

    Returns:
        tuple: (integral, step_size) where integral is the computed integral
               and step_size is the size of each interval.
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    integral = (h / 2) * np.sum(func(x[:-1]) + func(x[1:]))
    return integral, h


def int_simpson(func, a, b, n):
    """
    Simpson's rule for numerical integration.

    Parameters:
        func (callable): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of intervals (must be even).

    Returns:
        tuple: (integral, step_size) where integral is the computed integral
               and step_size is the size of each interval.
    """
    if n % 2 != 0:
        raise ValueError("Number of intervals (n) must be even for Simpson's rule.")

    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    integral = (h / 3) * (
        func(x[0])
        + 4 * np.sum(func(x[1:-1:2]))
        + 2 * np.sum(func(x[2:-2:2]))
        + func(x[-1])
    )
    return integral, h


def richardson_extrapolation(integration_method, func, a, b, n):
    """
    Richardson extrapolation for numerical integration.

    Parameters:
        integration_method (callable): Numerical integration method.
        func (callable): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of intervals.

    Returns:
        float: Improved integral value using Richardson extrapolation.
    """
    # Compute S(h) with n intervals
    S_h, _ = integration_method(func, a, b, n)

    # Compute S(h/2) with 2n intervals
    S_h2, _ = integration_method(func, a, b, 2 * n)

    # Apply Richardson extrapolation formula
    T_h = (4 * S_h2 - S_h) / 3

    return T_h
