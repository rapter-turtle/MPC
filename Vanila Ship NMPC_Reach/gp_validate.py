import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
def rbf_kernel(X1, X2, length_scale, variance):
    """
    Radial Basis Function (RBF) kernel (Gaussian kernel).
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)

def predictive_mean(X_new, X_train, alpha, length_scale, variance):
    """
    Compute the predictive mean of the GP at the new input points.
    """
    K_new = rbf_kernel(X_new, X_train, length_scale, variance)
    mu_new = np.dot(K_new, alpha)
    return mu_new

def predictive_variance(X_new, X_train, L, length_scale, variance, sigma_n):
    """
    Compute the predictive variance of the GP at the new input points.
    """
    K_new = rbf_kernel(X_new, X_train, length_scale, variance)
    v = np.linalg.solve(L, K_new.T)
    K_ss = rbf_kernel(X_new, X_new, length_scale, variance)
    var_new = np.diag(K_ss) - np.sum(v**2, axis=0)
    return var_new

filename = 'data_output2.txt'  # Adjust the filename if necessary
data = np.loadtxt(filename, delimiter='\t')

# Extract input features (x, y, z, dx, dy, dpsi) and target variable (V)
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

# Load model data from a .npz file
model_data = np.load('gp_model_data.npz')

# Extract the saved model components
X_train = model_data['X_train']
y_train = model_data['y_train']
L = model_data['L']
alpha = model_data['alpha']
length_scale = model_data['length_scale'].item()
variance = model_data['variance'].item()
sigma_n = model_data['sigma_n'][0]
K_train = model_data['K_train']




# Use a subset of the data for validation
subset_size = 1000 # Adjust the subset size as needed
indices = np.random.choice(X.shape[0], subset_size, replace=False)
X_subset = X[indices]
newV_subset = newV[indices]

# Compute the predictive mean and variance for the subset
start_time = time.time()
mu_new = predictive_mean(X_subset, X_train, alpha, length_scale, variance)
var_new = predictive_variance(X_subset, X_train, L, length_scale, variance, sigma_n)
end_time = time.time()

# print("Predicted means:", mu_new)
# print("Predicted variances:", var_new)
# print(f"Time taken for prediction: {end_time - start_time} seconds")


# rmse = np.sqrt(mean_squared_error(mu_new, newV_subset))
# nrmse = (rmse / (newV.max() - newV.min()))*100
# print("RMSE between the whole data and GP model:", nrmse, "%")




n2 = X_train.shape[0]
K = np.zeros(n2)
error = np.zeros(n2)

for j in range(subset_size):
    for i in range(n2):
        diff = X[indices[j]] - X_train[i, :]
        sqdist = np.dot(diff, diff)
        K[i] = variance * np.exp(-0.5 * sqdist / length_scale**2)
            
    mu = np.dot(K,K_train)

    # print( predictive_mean(X[indices[j]].reshape(1, -1), X_train, alpha, length_scale, variance))
    error[j] = mu - predictive_mean(X[indices[j]].reshape(1, -1), X_train, alpha, length_scale, variance)

rmse = np.sqrt(np.sum(error**2))
print("RMSE between the whole data and GP model:", rmse)


# Plotting
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Ground truth
# # ax.scatter(X_subset[:, 0], X_subset[:, 1], newV_subset, c='r', marker='o', label='Ground Truth')

# # GP predictions
# ax.scatter(X_subset[:, 0], X_subset[:, 1], mu_new-newV_subset, c='b', marker='^', label='GP Prediction')

# # Labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('newV')
# ax.set_title('Ground Truth and GP Predictions')
# ax.legend()

# plt.show()
