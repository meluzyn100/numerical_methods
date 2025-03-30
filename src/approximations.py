import numpy as np
from scipy.integrate import quad


def scalar_product(f, g, a, b, w):
    """
    Compute the weighted scalar product <f, g>_w on the interval [a, b].

    Parameters:
        f (callable): First function.
        g (callable): Second function.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        w (callable): Weight function w(x).

    Returns:
        float: The scalar product <f, g>_w.
    """

    def integrand(x):
        return f(x) * g(x) * w(x)

    result, _ = quad(integrand, a, b)
    return result


def least_squares_approximation(f, a, b, n, w=None, phi_k=None):
    """
    Approximate a function f(x) using the least squares method.

    Parameters:
        f (callable): The target function to approximate.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        n (int): Number of basis functions.
        w (callable, optional): Weight function w(x). Defaults to a constant weight of 1.
        phi_k (callable, optional): Basis function phi_k(x, k). Defaults to monomials x^k.

    Returns:
        np.ndarray: Coefficients of the approximating function in the chosen basis.
    """
    # Initialize the alpha matrix and beta vector
    alpha = np.zeros((n, n))
    beta = np.zeros(n)

    # Default weight function is constant (w(x) = 1)
    if w is None:

        def w(x):
            return 1

    # Default basis functions are monomials (phi_k(x, k) = x^k)
    if phi_k is None:

        def phi_k(x, k):
            return x**k

    # Compute the alpha matrix and beta vector
    for i in range(n):
        for j in range(n):
            alpha[i, j] = scalar_product(
                lambda x: phi_k(x, i), lambda x: phi_k(x, j), a, b, w
            )
        beta[i] = scalar_product(f, lambda x: phi_k(x, i), a, b, w)

    # Solve the linear system alpha * a = beta to find the coefficients
    a_coefficients = np.linalg.solve(alpha, beta)

    return a_coefficients


def approximating_function(x, a_coefficients, phi_k=None):
    """
    Evaluate the approximating function at a given point x.

    Parameters:
        x (float): The point at which to evaluate the approximating function.
        a_coefficients (np.ndarray): Coefficients of the approximating function.
        phi_k (callable, optional): Basis function phi_k(x, k). Defaults to monomials x^k.

    Returns:
        float: The value of the approximating function at x.
    """
    if phi_k is None:

        def phi_k(x, k):
            return x**k

    return sum(a_coefficients[k] * phi_k(x, k) for k in range(len(a_coefficients)))


def legendre_polynomial(n, x):
    """
    Compute the Legendre polynomial P_n(x) using an iterative approach.

    Parameters:
        n (int): Degree of the Legendre polynomial.
        x (float or np.ndarray): Input value(s) at which to evaluate the polynomial.

    Returns:
        float or np.ndarray: Value(s) of the Legendre polynomial P_n(x).
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x

    P0, P1 = np.ones_like(x), x
    for k in range(2, n + 1):
        P_next = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k
        P0, P1 = P1, P_next

    return P1


def orthonormal_legendre(x, n):
    """
    Generate the orthonormal Legendre polynomial function p_n(x).

    Parameters:
        n (int): Degree of the Legendre polynomial.

    Returns:
        callable: A function p_n(x) representing the orthonormal Legendre polynomial.
    """

    normalization_factor = np.sqrt((2 * n + 1) / 2)
    return legendre_polynomial(n, x) * normalization_factor
