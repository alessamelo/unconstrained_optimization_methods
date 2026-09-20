import numpy as np

def conjugate_gradient(x0, f, gradient, tol, max_iter, history):
    x_old = x0.copy().astype(float)
    history_data = []

    # Initial gradient
    grad_old = gradient(x_old[0], x_old[1])

    # Initial search direction
    direction_old = -grad_old

    for i in range(max_iter):

        # Backtracking line search
        alpha = 1.0
        c = 1e-4
        rho = 0.5

        f_old = f(x_old[0], x_old[1])
        directional_derivative = np.dot(grad_old, direction_old)

        while True:
            x_new = x_old + alpha * direction_old

            f_new = f(x_new[0], x_new[1])

            if f_new <= f_old + c * alpha * directional_derivative:
                break

            alpha *= rho

        # New gradient
        grad_new = gradient(x_new[0], x_new[1])

        # Step size
        r = np.linalg.norm(x_new - x_old)

        # Save iteration
        history(
            history_data,
            i + 1,
            x_new,
            f_new,
            grad_new,
            r
        )

        # Convergence
        if np.linalg.norm(grad_new) < tol:
            break

        # Fletcher-Reeves beta
        beta = (
            np.dot(grad_new, grad_new)
            / np.dot(grad_old, grad_old)
        )

        # New conjugate direction
        direction_new = -grad_new + beta * direction_old

        # Update
        x_old = x_new
        grad_old = grad_new
        direction_old = direction_new

    return x_new, f_new, i + 1, history_data