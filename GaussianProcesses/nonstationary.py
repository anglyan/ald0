import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

# Define the non-stationary function
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.squeeze(np.exp(0.3 * X) * np.sin(X))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

# Define the Locally Periodic Kernel (RBF * ExpSineSquared)
kernel = RBF(length_scale=1.0) * ExpSineSquared(length_scale=1.0, periodicity=3.0)


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)
gp.kernel_


mean_prediction, std_prediction = gp.predict(X, return_std=True)


plt.figure()
plt.plot(X,y, label = "True Function", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label = "Mean Prediction")
plt.fill_between(X.ravel(), mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha = 0.5, label = r"95% confidence interval",)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian Process Regression on Non-Stationary Function w/LPK")

plt.savefig("nonstationary_LP.png", dpi=300)
plt.clf()
# Print the optimized kernel parameters
# print("Optimized Kernel:", gp.kernel_)
