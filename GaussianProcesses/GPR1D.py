import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

X = np.linspace(start=0.0, stop=20, num=1000).reshape(-1,1)
y = np.squeeze(X * np.cos(X))

# Noise-free example
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# Plot noise-free GP regression
plt.figure()  # Create a new figure
plt.plot(X, y, label=r"$f(x) = x \cos(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean Prediction")
plt.fill_between(X.ravel(), mean_prediction - 1.96 * std_prediction, 
                 mean_prediction + 1.96 * std_prediction, alpha=0.5, label=r"95% confidence interval")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian Process Regression Noise-Free")
plt.legend()
plt.savefig("test.png", dpi=300)
plt.clf()

# With noise
noise_std = 0.85
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)
n_gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
n_gaussian_process.fit(X_train, y_train_noisy)
n_mean_prediction, n_std_prediction = n_gaussian_process.predict(X, return_std=True)

# Plot GP regression with noise
plt.plot(X, y, label=r"$f(x) = x \cos(x)$", linestyle="dotted")
plt.errorbar(X_train, y_train_noisy, noise_std, linestyle="None", color="tab:blue", marker=".", markersize=10, label="Noisy Observations")
plt.plot(X, n_mean_prediction, label="Mean Prediction with Noise")
plt.fill_between(X.ravel(), n_mean_prediction - 1.96 * n_std_prediction, 
                 n_mean_prediction + 1.96 * n_std_prediction, alpha=0.5, label=r"95% confidence interval")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian Process Regression With Noise")
plt.legend()
plt.savefig("test2.png", dpi=300)
plt.clf()  # Clear the figure after saving
