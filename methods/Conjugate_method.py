import numpy as np

def conjugate_gradient(x0, f, gradient, tol, max_iter, history):
    # Make a copy of the initial point so we don't modify the original
    x_old = x0.copy().astype(float)
    # Store the information from each iteration
    history_data = []
    # Calculate the gradient at the initial point
    grad_old = gradient(x_old[0], x_old[1])
    # Start by moving in the opposite direction of the gradient
    direction_old = -grad_old
    # Repeat the process until we reach the maximum number of iterations
    for i in range(max_iter):
        # Start with a step size of 1
        alpha = 1.0
        # Parameters used for the backtracking search
        c = 1e-4
        rho = 0.5

        # Get the function value at the current point
        f_old = f(x_old[0], x_old[1])

        # Check how much the function should decrease in the current direction
        directional_derivative = np.dot(grad_old, direction_old)

        # Reduce alpha until we find a step that decreases the function
        while True:
            # Try the new point using the current step size
            x_new = x_old + alpha * direction_old

            # Calculate the function value at the new point
            f_new = f(x_new[0], x_new[1])

            # Check the Armijo condition
            if f_new <= f_old + c * alpha * directional_derivative:
                break

            # If the condition is not satisfied, take a smaller step
            alpha *= rho
        # Calculate the gradient at the new point
        grad_new = gradient(x_new[0], x_new[1])
        # Calculate how much the point changed in this iteration
        r = np.linalg.norm(x_new - x_old)
        # Save the results of the current iteration
        history(history_data, i + 1, x_new, f_new, grad_new, r)
        # Stop if the gradient is small enough
        if np.linalg.norm(grad_new) < tol:
            break

        # Calculate beta using the Fletcher-Reeves formula
        beta = (
            np.dot(grad_new, grad_new)
            / np.dot(grad_old, grad_old)
        )

        # Use the new gradient to calculate the next conjugate direction
        direction_new = -grad_new + beta * direction_old
        # Move everything to the next iteration
        x_old = x_new
        grad_old = grad_new
        direction_old = direction_new

    # Return the final point, function value, number of iterations and history
    return x_new, f_new, i + 1, history_data