import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D


# sample data
np.random.seed(42)

fiberlength = np.random.uniform(5, 20, size=100) # mm
resinratio = np.random.uniform(0.1, 0.9, size=100) #fraction of weight
curingtemp = np.random.uniform(50, 200,  size=100) # C
pressure = np.random.uniform(1, 10, size=100) # MPa

# define the function we relate the variables with
material_strength = (
    0.5 * fiberlength * np.sin(resinratio * np.pi) +
    0.3 * np.log(curingtemp + 1) * pressure +
    20 * resinratio +
    np.random.normal(0, 1.5, 100)  # Adding some noise
)

X = np.column_stack((fiberlength, resinratio, curingtemp, pressure))
y = material_strength  # Target variable (Material Strength)




kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(X, y)


# Set fixed values for resin_ratio and curing_temp
fixed_resin_ratio = 0.5
fixed_curing_temp = 150

# Create a mesh grid for the other two parameters (fiber_length, pressure_applied)
fiber_length_range = np.linspace(5, 20, 50)
pressure_applied_range = np.linspace(1, 10, 50)
fiber_length_grid, pressure_applied_grid = np.meshgrid(fiber_length_range, pressure_applied_range)

# Flatten the grid and prepare input data with fixed values for prediction
X_pred = np.column_stack((
    fiber_length_grid.ravel(),
    np.full(fiber_length_grid.size, fixed_resin_ratio),
    np.full(fiber_length_grid.size, fixed_curing_temp),
    pressure_applied_grid.ravel()
))

# Predict the material strength on this grid
y_pred, y_std = gpr.predict(X_pred, return_std=True)
y_pred = y_pred.reshape(fiber_length_grid.shape)
y_std = y_std.reshape(fiber_length_grid.shape)

# Plot the mean predictions as a surface plot
fig = plt.figure(figsize=(14, 6))

# Mean prediction
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(fiber_length_grid, pressure_applied_grid, y_pred, cmap='viridis', edgecolor='k')
ax1.set_title("Mean Material Strength Prediction")
ax1.set_xlabel("Fiber Length (mm)")
ax1.set_ylabel("Pressure Applied (MPa)")
ax1.set_zlabel("Material Strength")
cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1, location='left')  # Move color bar left

# Uncertainty (Standard deviation)
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(fiber_length_grid, pressure_applied_grid, y_std, cmap='plasma', edgecolor='k')
ax2.set_title("Prediction Uncertainty")
ax2.set_xlabel("Fiber Length (mm)")
ax2.set_ylabel("Pressure Applied (MPa)")
ax2.set_zlabel("Uncertainty (Standard Deviation)")
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1, location='left')  # Move color bar left

plt.savefig("fourD.png")

