import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gradient_descent(x0, f, gradient, tol, max_iter, alpha, history):
    x_old = x0.copy()
    history_data = []
    for i in range(max_iter):
        grad = gradient(x_old[0], x_old[1])
        x_new = x_old - alpha * grad
        r = np.linalg.norm(x_new - x_old)
        history(history_data, i + 1, x_new, f(x_new[0], x_new[1]), grad, r)
        if r < tol:
            break
        x_old = x_new
    if i == max_iter-1:
        print("No convergence")
    return x_new, f(x_new[0], x_new[1]), i+1, history_data


