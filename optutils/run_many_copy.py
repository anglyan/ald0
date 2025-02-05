import os
import numpy as np
import pickle
from scipy.optimize import minimize_scalar
from aldfast import ALDdose, CostF, repeated_optimization
from optimize import optimize


def evaluate_cf(kernel_name: str, kernel_params: dict, a=0.1, b=20):
    conds = np.loadtxt("/home/yyardi/projects/ald0/optutils/ald_conds.dat")
    n_conds = conds.shape[0]

    optimized_t_values = []
    real_t_values = []
    num_iterations = []
    errors = []

    for i in range(n_conds):
        gpc = conds[i, 0]
        t0 = conds[i, 1]

        ald = ALDdose(t0, gpc)
        cf = CostF(ald, a, b)

        abs_min_t, abs_C_mins, local_mins, num_iter = optimize(cf)
        optimized_t_values.append(abs_min_t.item() if isinstance(abs_min_t, np.ndarray) else abs_min_t)
        num_iterations.append(num_iter)

        result = minimize_scalar(cf, bounds=(0.1, 20), method='bounded')
        if result.success:
            real_t_values.append(float(result.x))
        else:
            print(f"Optimization failed for condition {i}")

    # Calculate error metrics
    errors = [abs(opt - real) / real * 100 for opt, real in zip(optimized_t_values, real_t_values)]
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    avg_iterations = np.mean(num_iterations)

    # Save the results
    save_results(optimized_t_values, real_t_values, avg_error, std_error, avg_iterations, kernel_name, kernel_params)

    return optimized_t_values, real_t_values, avg_error, std_error, avg_iterations


def save_results(optimized_values, real_values, avg_error, std_error, avg_iterations, kernel_name, kernel_params):
    # Create a directory for organized storage
    save_dir = "/home/yyardi/projects/ald0/optutils/results"
    os.makedirs(save_dir, exist_ok=True)

    # Create a filename based on kernel name and hyperparameters
    param_str = "_".join(f"{k}={v}" for k, v in kernel_params.items())
    filename = f"{kernel_name}_{param_str}.pkl"

    # Save all key metrics in the pickle
    results = {
        "optimized_t_values": optimized_values,
        "real_t_values": real_values,
        "avg_error": avg_error,
        "std_error": std_error,
        "avg_iterations": avg_iterations,
    }

    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    kernel_name = "Matern"  # Specify the kernel
    kernel_params = {"length_scale": 1.0, "length_scale_bounds": (0.1, 10.0), "nu": 2.5}  # Parameters for Matern
    optimized_values, real_values, avg_error, std_error, avg_iterations = evaluate_cf(kernel_name, kernel_params)
    print(f"Results saved for kernel {kernel_name} with params {kernel_params}")
    print(f"Avg Error: {avg_error}%, Std Error: {std_error}%, Avg Iterations: {avg_iterations}")
