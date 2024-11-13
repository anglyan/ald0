import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize


def get_samples(start, end, step):
    """Generate an array of t values from start to end with a given step size."""
    return np.arange(start, end, step)

def optimize(cf, t_range=(0.1, 10), n_samples=15, max_iter=20, tolerance=1e-3):
    """
    Minimize the cost function using Bayesian Optimization.
    
    Parameters:
    - cf: Callable cost function.
    - t_range: Tuple with the initial range for t (min_t, max_t).
    - n_samples: Number of samples in each iteration to start with.
    - max_iter: Maximum iterations for Bayesian Optimization.
    - tolerance: Stopping criterion based on the improvement in minimum.
    
    Returns:
    - t_min: Optimal t value.
    - C_min: Minimum cost function value at t_min.
    """
    min_cost = np.inf
    t_min = None

    for _ in range(max_iter):
        # generate sample points
        t_samples = get_samples(t_range[0], t_range[1], (t_range[1] - t_range[0]) / n_samples)
        C_samples = []
        
        for t in t_samples:
            try:
                C = cf(t)
                if np.isnan(C) or np.isinf(C):
                    C = 1e6  # High penalty for NaN or Inf values
                C_samples.append(C)
            except ValueError:
                C_samples.append(1e6)  # High penalty if t <= 0

        # Step 3.2: Run Bayesian Optimization
        def objective(x):
            try:
                C = cf(x[0])
                return C if not (np.isnan(C) or np.isinf(C)) else 1e6
            except ValueError:
                return 1e6

        result = gp_minimize(objective, [t_range], n_calls=n_samples, random_state=42)
        
        # Step 3.3: Check convergence
        if result.fun < min_cost - tolerance:
            min_cost = result.fun
            t_min = result.x[0]
            # Narrow down t_range for refined search around t_min
            t_range = (max(t_min - 0.5, 0.1), t_min + 0.5)
        else:
            break

    return t_min, min_cost



