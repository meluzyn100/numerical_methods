import numpy as np


def numerical_derivative(f, x, y, h):
    """
    Compute the numerical derivative of a function using the central difference method.

    Parameters:
        f (callable): The function to differentiate.
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        h (float): The step size for the finite difference.

    Returns:
        float: The numerical derivative.
    """
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def euler_method(f, y0, x0, x_end, step_size):
    """
    Solve an ODE using the explicit Euler method.

    Parameters:
        f (callable): The derivative function f(x, y).
        y0 (float): The initial value of y.
        x0 (float): The initial value of x.
        x_end (float): The final value of x.
        step_size (float): The step size for the method.

    Returns:
        tuple: Arrays of x and y values.
    """
    x_values = np.arange(x0, x_end + step_size, step_size)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for i in range(1, len(x_values)):
        y_values[i] = y_values[i - 1] + step_size * \
            f(x_values[i - 1], y_values[i - 1])

    return x_values, y_values


def backward_euler_method(f, y0, x0, x_end, step_size, tol=1e-6, max_iter=100):
    """
    Solve an ODE using the implicit (backward) Euler method.

    Parameters:
        f (callable): The derivative function f(x, y).
        y0 (float): The initial value of y.
        x0 (float): The initial value of x.
        x_end (float): The final value of x.
        step_size (float): The step size for the method.
        tol (float): Tolerance for the Newton-Raphson iteration.
        max_iter (int): Maximum number of iterations for Newton-Raphson.

    Returns:
        tuple: Arrays of x and y values.
    """
    n = int((x_end - x0) / step_size) + 1
    x_values = np.linspace(x0, x_end, n)
    y_values = np.zeros(n)
    y_values[0] = y0

    for i in range(1, n):
        x_next = x_values[i]
        y_current = y_values[i - 1]
        y_next_guess = y_current

        # Newton-Raphson iteration to solve the implicit equation
        for _ in range(max_iter):
            g = y_next_guess - y_current - step_size * f(x_next, y_next_guess)
            df_dy = numerical_derivative(f, x_next, y_next_guess, step_size)
            g_prime = 1 - step_size * df_dy
            y_next_new = y_next_guess - g / g_prime

            if abs(y_next_new - y_next_guess) < tol:
                y_next_guess = y_next_new
                break
            y_next_guess = y_next_new

        y_values[i] = y_next_guess

    return x_values, y_values


def runge_kutta_2nd_order(f, y0, x0, x_end, step_size):
    """
    Solve an ODE using the second-order Runge-Kutta method.

    Parameters:
        f (callable): The derivative function f(x, y).
        y0 (float): The initial value of y.
        x0 (float): The initial value of x.
        x_end (float): The final value of x.
        step_size (float): The step size for the method.

    Returns:
        tuple: Arrays of x and y values.
    """
    x_values = np.arange(x0, x_end + step_size, step_size)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for i in range(1, len(x_values)):
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]
        k1 = f(x_n, y_n) * step_size
        k2 = f(x_n + step_size, y_n + k1) * step_size
        y_values[i] = y_n + 0.5 * (k1 + k2)

    return x_values, y_values


def runge_kutta_4th_order(f, y0, x0, x_end, step_size):
    """
    Solve an ODE using the fourth-order Runge-Kutta method.

    Parameters:
        f (callable): The derivative function f(x, y).
        y0 (float): The initial value of y.
        x0 (float): The initial value of x.
        x_end (float): The final value of x.
        step_size (float): The step size for the method.

    Returns:
        tuple: Arrays of x and y values.
    """
    x_values = np.arange(x0, x_end + step_size, step_size)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for i in range(1, len(x_values)):
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]
        k1 = f(x_n, y_n) * step_size
        k2 = f(x_n + 0.5 * step_size, y_n + 0.5 * k1) * step_size
        k3 = f(x_n + 0.5 * step_size, y_n + 0.5 * k2) * step_size
        k4 = f(x_n + step_size, y_n + k3) * step_size
        y_values[i] = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values, y_values


def finite_difference_method(a, b, N, q, g, alpha, beta):
    """
    Solve the boundary value problem:
        -
    y(a) = alpha
    y(b) = beta
    using the finite difference method.

    Parameters:
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        N (int): Number of subintervals.
        q (function): Function q(x).
        g (function): Function g(x).
        alpha (float): Boundary condition at x = a.
        beta (float): Boundary condition at x = b.

    Returns:
        x (numpy.ndarray): Discretized x values.
        y (numpy.ndarray): Approximate solution at the discretized points.
    """
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    A[0, 0] = 1
    A[N, N] = 1
    F[0] = alpha
    F[N] = beta

    for i in range(1, N):
        xi = x[i]
        A[i, i - 1] = -1
        A[i, i] = 2 + q(xi) * h**2
        A[i, i + 1] = -1
        F[i] = g(xi) * h**2

    y = np.linalg.solve(A, F)
    return x, y


def solve_heat_equation(
    L, T, n, m, diffusion_coeff, boundary_left, boundary_right, initial_condition
):
    """
    Solve the heat equation using the Crank-Nicolson method.

    Parameters:
        L (float): Length of the spatial domain.
        T (float): Total simulation time.
        n (int): Number of spatial divisions.
        m (int): Number of time steps.
        diffusion_coeff (float): Diffusion coefficient.
        boundary_left (float): Boundary condition at x = 0.
        boundary_right (float): Boundary condition at x = L.
        initial_condition (callable): Function defining the initial condition f(x).

    Returns:
        tuple:
            - xn (numpy.ndarray): Spatial grid points.
            - tm (numpy.ndarray): Time grid points.
            - u (numpy.ndarray): Solution matrix where u[i, j] is the solution at x_i and t_j.
    """
    # Spatial and temporal step sizes
    dx = L / n
    dt = T / m

    # Stability parameter
    lambda_ = diffusion_coeff * dt / dx**2

    # Check stability condition
    if lambda_ >= 0.5:
        raise ValueError(
            f"Stability condition violated: lambda must be less than 0.5. Given: lambda = {lambda_:.2f}"
        )

    # Discretize spatial and temporal domains
    xn = np.linspace(0, L, n + 1)
    tm = np.linspace(0, T, m + 1)

    # Initialize solution matrix
    u = np.zeros((n + 1, m + 1))
    u[:, 0] = initial_condition(xn)

    # Apply boundary conditions
    u[0, :] = boundary_left
    u[-1, :] = boundary_right

    # Construct matrices for Crank-Nicolson method
    main_diag_A = (1 + lambda_) * np.ones(n - 1)
    off_diag_A = -lambda_ / 2 * np.ones(n - 2)
    A = np.diag(main_diag_A) + np.diag(off_diag_A, k=1) + \
        np.diag(off_diag_A, k=-1)

    main_diag_B = (1 - lambda_) * np.ones(n - 1)
    off_diag_B = lambda_ / 2 * np.ones(n - 2)
    B = np.diag(main_diag_B) + np.diag(off_diag_B, k=1) + \
        np.diag(off_diag_B, k=-1)

    # Time-stepping loop
    for j in range(m):
        b = B @ u[1:-1, j]
        u[1:-1, j + 1] = np.linalg.solve(A, b)

    return xn, tm, u
