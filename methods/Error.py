import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def history(history_data, iteration, x, f_value, grad, error):
    grad_norm = np.linalg.norm(grad)
    history_data.append({
        "iteration": iteration,
        "x1": x[0],
        "x2": x[1],
        "f(x)": f_value,
        "gradient_x1": grad[0],
        "gradient_x2": grad[1],
        "gradient_norm": grad_norm,
        "error": error
    })

    
def save_iterations(history, method, x_true):
    results_folder = "results"
    plots_folder = os.path.join(results_folder, "plots")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    df = pd.DataFrame(history)
    df["true_error"] = df.apply(
        lambda row: np.linalg.norm(
            np.array([row["x1"], row["x2"]]) - x_true
        ),
        axis=1
    )
    csv_path = os.path.join(results_folder, f"{method}.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="iteration",
        y="true_error",
        marker="o"
    )
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|x_k - x^*\|$")
    plt.title(f"{method}: True Error vs Iteration")
    plt.grid(True)

    plot_path = os.path.join(
        plots_folder,
        f"{method}_error.png"
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    return df



def plot_methods_error(results_folder, plots_folder=None):
    if plots_folder is None:
        plots_folder = os.path.join(results_folder, "plots")

    os.makedirs(plots_folder, exist_ok=True)

    files = {
        "Steepest Descent": "Gradient_descent.csv",
        "Conjugate Gradient": "Conjugate Gradient Method.csv",
        "Newton Method": "Newton Method.csv"
    }

    styles = {
        "Steepest Descent": {"marker": "o"},
        "Conjugate Gradient": {"marker": "s"},
        "Newton Method": {"marker": "^"}
    }

    data = {}

    for method, filename in files.items():
        path = os.path.join(results_folder, filename)
        data[method] = pd.read_csv(path)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # --------------------------------------------------
    # Complete convergence
    # --------------------------------------------------
    for method, df in data.items():
        axes[0].plot(
            df["iteration"],
            df["true_error"],
            marker=styles[method]["marker"],
            markersize=4,
            linewidth=1.5,
            label=method
        )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\|x_k - x^*\|$")
    axes[0].set_title("Complete Convergence")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # --------------------------------------------------
    # First 50 iterations
    # --------------------------------------------------
    zoom_iterations = 50

    for method, df in data.items():
        df_zoom = df[df["iteration"] <= zoom_iterations]

        axes[1].plot(
            df_zoom["iteration"],
            df_zoom["true_error"],
            marker=styles[method]["marker"],
            markersize=5,
            linewidth=1.5,
            label=method
        )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|x_k - x^*\|$")
    axes[1].set_title(f"Convergence: First {zoom_iterations} Iterations")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1, zoom_iterations)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    plot_path = os.path.join(
        plots_folder,
        "methods_comparison_error.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Plot saved:", plot_path)



def calculate_precision(results_folder):
    files = {
        "Steepest Descent": "Gradient_descent.csv",
        "Conjugate Gradient": "Conjugate Gradient Method.csv",
        "Newton Method": "Newton Method.csv"
    }

    results = []

    for method, filename in files.items():
        path = os.path.join(results_folder, filename)

        df = pd.read_csv(path)

        final_error = abs(df["true_error"].iloc[-1])

        if final_error == 0:
            decimals = np.inf
        else:
            decimals = max(
                0,
                int(np.floor(-np.log10(2 * final_error)))
            )

        results.append({
            "Method": method,
            "Iterations": len(df),
            "Final Error": final_error,
            "Correct Decimals": decimals
        })

    precision_df = pd.DataFrame(results)

    output_path = os.path.join(
        results_folder,
        "methods_precision.csv"
    )

    precision_df.to_csv(
        output_path,
        index=False
    )

    print(precision_df)
    print("\nDatabase saved:", output_path)

    return precision_df

def ten_iterations(results_folder):
    files = {
        "Steepest Descent": "Gradient_descent.csv",
        "Conjugate Gradient": "Conjugate Gradient Method.csv",
        "Newton Method": "Newton Method.csv"
    }

    results = []

    for method, filename in files.items():
        path = os.path.join(results_folder, filename)

        df = pd.read_csv(path)

        # Take only the first 10 iterations
        df = df.head(10).copy()

        # Add method name
        df.insert(0, "Method", method)

        results.append(df)

    # Combine all methods
    iterations_df = pd.concat(
        results,
        ignore_index=True
    )

    # Save database
    output_path = os.path.join(
        results_folder,
        "first_10_iterations.csv"
    )

    iterations_df.to_csv(
        output_path,
        index=False
    )

    print(iterations_df)
    print("\nDatabase saved:", output_path)

    return iterations_df