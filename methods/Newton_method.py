import numpy as np


def newton_method(x0, f, gradient, hessian, tol, max_iter, alpha, history):
    # Make a copy of the initial point
    x_old = x0.copy()

    # Store the information from each iteration
    history_data = []

    # Repeat the method until convergence or the maximum number of iterations
    for i in range(max_iter):

        # Calculate the gradient at the current point
        grad = gradient(x_old[0], x_old[1])

        # Calculate the Hessian matrix at the current point
        H = hessian(x_old[0], x_old[1])

        # Find the Newton direction by solving H * d = -gradient
        d = np.linalg.solve(H, -grad)

        # Move to the new point using the Newton direction
        x_new = x_old + d

        # Calculate how much the point changed
        r = np.linalg.norm(x_new - x_old)

        # Save the results from this iteration
        history(history_data,i + 1, x_new, f(x_new[0], x_new[1]),grad,r)
                                                    
        # Stop if the change in the point is small enough
        if r < tol:
            break

        # Use the new point for the next iteration
        x_old = x_new

    # Return the final point, function value, number of iterations and history
    return x_new, f(x_new[0], x_new[1]), i + 1, history_data