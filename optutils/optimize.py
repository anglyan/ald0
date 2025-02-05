import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import os
path = "/home/yyardi/projects/ald0/optutils/images"

def get_samples(start, end, step):
    """Generate an array of t values from start to end with a given step size."""
    return np.arange(start, end, step)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.1):
    """
    Calculate the Expected Improvement (EI) acquisition function.
    
    Parameters:
    - X: Points where EI should be evaluated (array-like, shape: (n_points, 1)).
    - X_sample: Sampled input points (array-like, shape: (n_samples, 1)).
    - Y_sample: Corresponding function values at X_sample (array-like, shape: (n_samples,)).
    - gpr: A fitted GaussianProcessRegressor.
    - xi: Exploration-exploitation trade-off parameter.
    
    Returns:
    - EI values for each point in X.
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.min(mu_sample)  # Current best observed value

    with np.errstate(divide='warn'):
        Z = (mu_sample_opt - mu - xi) / sigma
        ei = (mu_sample_opt - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # Handle zero variance edge case

    return ei

def propose_location(X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """
    Propose the next sampling point by optimizing the acquisition function.
    
    Parameters:
    - X_sample: Sampled input points (array-like, shape: (n_samples, 1)).
    - Y_sample: Corresponding function values at X_sample (array-like, shape: (n_samples,)).
    - gpr: A fitted GaussianProcessRegressor.
    - bounds: Bounds for the search space as (min, max).
    - n_restarts: Number of random starting points for the optimizer.
    
    Returns:
    - X_next: Proposed next point to sample (array-like, shape: (1,)).
    """
    dim = X_sample.shape[1]
    min_val = float("inf")
    X_next = None

    # Evaluate acquisition function at several random starting points
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[0], bounds[1], size=dim)
        res = minimize(
            lambda x: -expected_improvement(x.reshape(-1, 1), X_sample, Y_sample, gpr),
            x0=x0,
            bounds=[bounds],
            method="L-BFGS-B"
        )
        if res.fun < min_val:
            min_val = res.fun
            X_next = res.x

    return np.clip(X_next, bounds[0], bounds[1])

def optimize(cf, t_range=(0.1, 20), n_samples=6, max_iter=2, tolerance=1e-6):
    """
    Manual Bayesian Optimization for minimizing a cost function.

    Parameters:
    - cf: Callable cost function.
    - t_range: Tuple with the initial range for t (min_t, max_t).
    - n_samples: Number of initial random samples.
    - max_iter: Maximum iterations for Bayesian Optimization.
    - tolerance: Stopping criterion based on improvement in minimum.

    Returns:
    - t_min: Optimal t value.
    - C_min: Minimum cost function value at t_min.
    - local_mins: List of tuples (t, C) of all local minima observed.
    """
    # X_sample = np.random.uniform(t_range[0], t_range[1], size=(n_samples, 1))
    # X_sample = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_samples).reshape(-1, 1)
    # Y_sample = np.array([cf(t) for t in X_sample])
    X_sample = np.array([0.1,0.5,1,3,8,20]).reshape(-1, 1)
    Y_sample = np.array([cf(t) for t in X_sample])
    # x_additional = np.array([0.1, 20])
    # y_additional = np.array([cf(t) for t in x_additional])
    # X_sample = np.vstack((X_sample, x_additional.reshape(-1, 1))) # needs to be a 2D vector
    # Y_sample = np.append(Y_sample, y_additional)


    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    num_iteration_used = 0
    local_mins = []
    for iteration in range(max_iter):
        gpr.fit(X_sample, Y_sample)
        num_iteration_used += 1 
        # After first iteration, plot the current optimization progress with uncertainty
        if iteration == 0:
            plot_optimization_progress(cf, X_sample, Y_sample, t_range, gpr, iteration)

        X_next = propose_location(X_sample, Y_sample, gpr, t_range).reshape(1, -1)

        Y_next = cf(X_next[0, 0])
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.append(Y_sample, Y_next)

        # Track local minimum
        local_mins.append((X_next[0, 0], Y_next))
        print(f"Iteration {iteration + 1}: t = {X_next[0, 0]:.4f}, C = {Y_next:.4f}")

        if np.abs(np.min(Y_sample) - Y_next) < tolerance:
            break

    t_min = X_sample[np.argmin(Y_sample)]
    C_min = np.min(Y_sample)

    return t_min, C_min, local_mins, num_iteration_used


def plot_optimization_progress(cf, X_sample, Y_sample, t_range, gpr, iteration):
    """
    Plot the optimization progress with uncertainty (confidence intervals).
    """
    t_values = np.linspace(1, 10, 100).reshape(-1, 1)
    C_values = np.array([cf(t) for t in t_values])

    # Predict the mean and standard deviation from the Gaussian Process
    mu, sigma = gpr.predict(t_values, return_std=True)

    # Plot the cost function, GP predictions, uncertainty, and sampled points
    plt.plot(t_values, C_values, label="Original Cost Function", color='blue')
    plt.plot(t_values, mu, label="GP Mean", color='green')
    plt.fill_between(t_values.flatten(), mu - 1.96 * sigma, mu + 1.96 * sigma, color='gray', alpha=0.2, label="95% Confidence Interval")
    plt.scatter(X_sample, Y_sample, color='red', label="Sampled Points")
    plt.xlabel("Time (t)")
    plt.ylabel("Cost")
    plt.title(f"Optimization Progress after Iteration {iteration + 1}")
    plt.legend()

    plt.savefig(os.path.join(path, f"optimization_progress_iteration_{iteration + 1}.png"), dpi=300)
    plt.clf()