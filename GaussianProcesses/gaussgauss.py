import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the Gaussian function
def gauss(x, y, x0=0.5, y0=0.5, sx=0.1, sy=0.1):
    t2 = np.power((x - x0) / sx, 2) + np.power((y - y0) / sy, 2)
    return np.exp(-t2)

# Generate random sample data
x = np.random.random(size=500)
y = np.random.random(size=500)
out = gauss(x, y)

# training data
X_train = np.vstack((x, y)).T
y_train = out

# GPR 
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(X_train, y_train)

# plotting slices
x_line = np.linspace(0, 1, 100)
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

y_values = [0.2, 0.5, 0.8]  # Fixed y values for slices
for i, y_fixed in enumerate(y_values):
    y_true = gauss(x_line, y_fixed)

    X_pred = np.vstack([x_line, np.full_like(x_line, y_fixed)]).T
    y_pred, sigma = gpr.predict(X_pred, return_std=True)
    
    ax[i].plot(x_line, y_true, 'b-', label="True Gaussian")
    ax[i].plot(x_line, y_pred, 'r--', label="GPR Prediction")
    ax[i].fill_between(x_line, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='gray', alpha=0.3, label="95% Confidence Interval")
    ax[i].set_title(f"Cross-section at y = {y_fixed}")
    ax[i].legend()

plt.xlabel("x")
plt.ylabel("Function Value")
plt.savefig("gaussgauss.png")
