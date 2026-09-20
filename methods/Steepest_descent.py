import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gradient_descent(x0, f, gradient, tol, max_iter, alpha, history):
    # Make a copy of the initial point
    x_old = x0.copy()
    # Store the results from each iteration
    history_data = []
    for i in range(max_iter):
        # Calculate the gradient at the current point
        grad = gradient(x_old[0], x_old[1])
        # Move in the opposite direction of the gradient
        x_new = x_old - alpha * grad
        # Calculate the change between the old and new points
        r = np.linalg.norm(x_new - x_old)
        # Save the results from this iteration
        history(history_data, i + 1, x_new, f(x_new[0], x_new[1]), grad, r)
        # Stop if the change is small enough
        if r < tol:
            break
        # Use the new point for the next iteration
        x_old = x_new
    # Show a message if the method did not converge
    if i == max_iter - 1:
        print("No convergence")
    # Return the final point, function value, number of iterations and history
    return x_new, f(x_new[0], x_new[1]), i + 1, history_data