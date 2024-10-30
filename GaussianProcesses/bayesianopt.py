import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence

# Define the objective function
def objective(x):
    return (x[0] - 2)**2 + np.sin(5 * x[0])

# Define the bounds of the function
bounds = [(-2.0, 8.0)]

# Run Bayesian Optimization with Gaussian Processes
result = gp_minimize(objective, bounds, n_calls=30, random_state=42)

# Plot the result
x = np.linspace(-2, 8, 1000).reshape(-1, 1)
y = (x - 2)**2 + np.sin(5 * x)
plt.figure()
plt.plot(x, y, label="True Function")
plt.scatter(result.x_iters, result.func_vals, c='red', s=50, zorder=10, edgecolor='k', label="Samples")
plt.plot(result.x, result.fun, 'ro', label="Minimum found", markersize=12)
plt.legend()
plt.title("Bayesian Optimization with GP")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.savefig("bayesian_min.png", dpi=300)

# Plot the convergence (function values over iterations)
plt.clf()
plot_convergence(result)
plt.savefig("convergence.png", dpi=300)

