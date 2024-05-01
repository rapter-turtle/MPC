import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D module for 3D plotting


def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions
xyz0 = [0.001, 0.0, 0.0]

# Time span
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

# Solve the system of differential equations
sol = solve_ivp(lorenz, t_span, xyz0, args=(sigma, rho, beta), t_eval=t_eval)

# Extract solution
x, y, z = sol.y

# Plotting the Lorenz attractor
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'b-', linewidth=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()
