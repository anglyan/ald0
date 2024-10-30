import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C

# making the data
temperature = np.linspace(15, 30, 10)
humidity = np.linspace(30, 90, 10) 

temperature_grid, humidity_grid = np.meshgrid(temperature, humidity)
temperature_flat = temperature_grid.ravel()
humidity_flat = humidity_grid.ravel()


# optimize the air quality inside a smart greenhouse by adjusting two factors: temperature and humidity. 
# we are given the function y=sin(0.1⋅temperature)⋅cos(0.1⋅humidity)+random noise

def model(t, h):
    return ((np.sin(0.1*t))*(np.cos(0.2*h)) + np.random.normal(0, 0.05, t.shape))

grv = model(temperature_flat, humidity_flat)

grg = grv.reshape(temperature_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(temperature_grid, humidity_grid, grg, cmap='viridis')
plt.colorbar(label='Growth Rate Index')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Simulated Growth Rate Index by Temperature and Humidity')
plt.savefig("datagrid.png", dpi=300)
plt.clf()

X = np.column_stack((temperature_flat, humidity_flat))
y = grv  # Target variable (growth rate index)

kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.05**2)

gp.fit(X, y)

temperature_pred = np.linspace(15, 30, 40)
humidity_pred = np.linspace(30, 90, 40)
temperature_grid_pred, humidity_grid_pred = np.meshgrid(temperature_pred, humidity_pred)
X_pred = np.column_stack((temperature_grid_pred.ravel(), humidity_grid_pred.ravel()))

y_pred, sigma = gp.predict(X_pred, return_std=True)

y_pred_grid = y_pred.reshape(temperature_grid_pred.shape)
sigma_grid = sigma.reshape(temperature_grid_pred.shape)

# Plot the mean predicted growth rate index as a contour plot
plt.contourf(temperature_grid_pred, humidity_grid_pred, y_pred_grid, cmap='viridis')
plt.colorbar(label='Predicted Growth Rate Index')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Predicted Growth Rate Index by Temperature and Humidity')
plt.savefig("predictedgrid.png")
plt.clf()
# Plot the uncertainty (standard deviation) as a separate contour plot

plt.contourf(temperature_grid_pred, humidity_grid_pred, sigma_grid, cmap='coolwarm')
plt.colorbar(label='Prediction Uncertainty (σ)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Uncertainty of Prediction by Temperature and Humidity')
plt.savefig("uncertaintygrid.png")



