import numpy as np

def newton_method(x0, f, gradient, hessian, tol, max_iter, alpha, history):
    x_old = x0.copy()
    history_data = []

    for i in range(max_iter):
        grad = gradient(x_old[0], x_old[1])
        H = hessian(x_old[0], x_old[1])

        # Newton direction
        d = np.linalg.solve(H, -grad)

        # New point
        x_new = x_old + d

        # Step size
        r = np.linalg.norm(x_new - x_old)

        # Save iteration
        history(history_data, i + 1, x_new, f(x_new[0], x_new[1]), grad, r)

        if r < tol:
            break

        x_old = x_new

    return x_new, f(x_new[0], x_new[1]), i + 1, history_data