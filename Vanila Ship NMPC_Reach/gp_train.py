import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error

# Print environment information
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Installed packages path:", sys.path)

# Load data from the text file
filename = 'data_output2.txt'  # Adjust the filename if necessary
data = np.loadtxt(filename, delimiter='\t')

# Extract input features (x, y, z, dx, dy, dpsi) and target variable (V)
# X = data[:, [0,1,2,4,5,6]]  # Includes x, y, z, dx, dy, dpsi
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
dx = data[:, 4]
dy = data[:, 5]
dpsi = data[:, 6]
Vr = np.linspace(-3, 3, len(x))
r = np.linspace(-2, 2, len(x))
X = np.column_stack((x, y, z, Vr, r))


# Define the new target variable
newV = Vr * (-dx * np.cos(z) - dy * np.sin(z)) - r * dpsi + abs(dx) + abs(dy) + 0.05 * abs(dpsi)


# Use a subset of the data for training
subset_size = 1000  # Adjust the subset size as needed
indices = np.random.choice(X.shape[0], subset_size, replace=False)

X_subset = X[indices]
newV_subset = newV[indices]

# Define the kernel for the Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

# Create and fit the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
gp.fit(X_subset, newV_subset)

# Print the learned kernel and log-marginal-likelihood
print("Learned kernel:", gp.kernel_)
print("Log-Marginal-Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

# Predict the values for the entire dataset
newV_pred_full, sigma_full = gp.predict(X, return_std=True)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(newV, newV_pred_full))
nrmse = rmse / (newV.max() - newV.min())
print("RMSE between the whole data and GP model:", nrmse)

learned_kernel = gp.kernel_
learned_params = learned_kernel.get_params()
alpha = gp.alpha_
L = gp.L_
X_train = gp.X_train_
y_train = gp.y_train_

# Extract the hyperparameters from the learned kernel
length_scale = learned_kernel.k2.length_scale
variance = learned_kernel.k1.constant_value
sigma_n = gp.alpha_

K_train = np.zeros((subset_size, subset_size))
for i in range(subset_size):
    for j in range(subset_size):
        diff = X_train[i,:] - X_train[j, :]
        sqdist = np.dot(diff, diff)
        K_train[i, j] = variance * np.exp(-0.5 * sqdist / length_scale**2)
K_train = np.dot(np.linalg.inv(K_train), y_train)
# print(K_train)

# Save the model data to a .npz file
np.savez('gp_model_data.npz',
         X_train=X_train,
         y_train=y_train,
         L=L,
         alpha=alpha,
         length_scale=length_scale,
         variance=variance,
         sigma_n=sigma_n,
         K_train = K_train
         )


# # Plotting
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Ground truth
# ax.scatter(X_subset[:, 0], X_subset[:, 1], newV_subset, c='r', marker='o', label='Ground Truth')

# # GP predictions
# ax.scatter(X_subset[:, 0], X_subset[:, 1], newV_pred_full[indices], c='b', marker='^', label='GP Prediction')

# # Labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('newV')
# ax.set_title('Ground Truth and GP Predictions')
# ax.legend()

# plt.show()
