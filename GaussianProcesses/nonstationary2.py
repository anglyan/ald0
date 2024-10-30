import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Define the non-stationary function
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.squeeze(np.exp(0.3 * X) * np.sin(X))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel2 = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=10)
gp2.fit(X_train, y_train)
gp2.kernel_ #this is useless
mean_prediction2, std_prediction2 = gp2.predict(X, return_std=True)


plt.plot(X,y, label = "True Function", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction2, label = "Mean Prediction")
plt.fill_between(X.ravel(), mean_prediction2 - 1.96 * std_prediction2, mean_prediction2 + 1.96 * std_prediction2, alpha = 0.5, label = r"95% confidence interval",)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian Process Regression on Non-Stationary Function w/RBF")

plt.savefig("nonstationary_RBF.png", dpi=300)
plt.clf()