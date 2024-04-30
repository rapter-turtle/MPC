import numpy as np
import math
import matplotlib.pyplot as plt

def dynamics(x, u, t):
    x1dot = 10*(x[1] - x[0])
    x2dot = 28*x[0] - x[1] - x[0]*x[2] + (x[0]**2 + x[1]**2 + 0.1)*(u + 0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])+1)
    # x2dot = 0.5*x[0] - x[1] - x[0]*x[2] + (x[0]**2 + x[1]**2 + 0.1)
    print(x2dot)
    x3dot = x[0]*x[1] - 2.667*x[2]

    return np.array([x1dot, x2dot, x3dot])

def estim_dynamics(x, u ,x_error, param_estim):

    x1dot = 10*(x[1] - x[0]) - x_error[0]
    x2dot = 28*x[0] - x[1] - x[0]*x[2] + (x[0]**2 + x[1]**2 + 0.1)*(u + param_estim[0])  - x_error[1]
    x3dot = x[0]*x[1] - 2.667*x[2] - x_error[2]

 
    return np.array([x1dot, x2dot, x3dot])

def h_function(x, eta, x_max):
    h = (x*x - x_max*x_max)/(eta*x_max*x_max)
    hdot = (2*x)/(eta*x_max*x_max)
    return h, hdot

def param_dynamics(x ,x_error):
    
    P = np.array([[0.5, 0, 0],
              [0, 0.5, 0],
              [0, 0, 0.5]])

    g = np.array([0, x[0]**2 + x[1]**2 + 0.1 , 0])

    param_update = -np.dot(np.dot(g, P), x_error)
    

    h, hdot = h_function(param_update, 0.1, 0.1/math.sqrt(1+0.1))

    if h > 0 and param_update*hdot > 0:
        param_dot = 1e8*param_update*(1 - h)
    else:
        param_dot = param_update

    return param_dot


def simulate(x0, u, dt, num_steps):
    x = np.zeros((num_steps+1, len(x0)))
    param_estim = np.zeros((num_steps+1, 1))
    state_estim = np.zeros((num_steps+1, len(x0)))
    x[0] = x0
    param_estim[0] = 0
    state_estim[0] = x0

    cutoff = 70
    filtered_param = 0
    filtered_param_dot = 0

    for i in range(num_steps):
        t = i * dt

        # filtered_param = 0
        x[i+1] = x[i] + dt * dynamics(x[i], filtered_param, t)
        #######################################################################################################################
        state_estim[i+1] = state_estim[i] + dt*estim_dynamics(x[i], filtered_param, state_estim[i] - x[i], param_estim[i])
        param_estim[i+1] = param_estim[i] + dt*param_dynamics(x[i] ,state_estim[i] - x[i])
        
        u_dot = param_dynamics(x[i+1] ,state_estim[i+1] - x[i+1])
        filtered_param_dot = filtered_param_dot + dt * (-cutoff * filtered_param_dot + cutoff * u_dot)
        filtered_param = filtered_param + dt * filtered_param_dot
        #######################################################################################################################


        
    return x, param_estim, state_estim

def plot_simulation(x, param_estim, state_estim, dt):
    num_steps = len(x) - 1
    time = np.arange(0, num_steps * dt + dt, dt)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, x[:, 0], label='x1')
    plt.plot(time, x[:, 1], label='x2')
    plt.plot(time, x[:, 2], label='x3')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Simulation')
    plt.legend()
    plt.grid(True)
    
    num_steps = len(param_estim)-1 
    time = np.arange(0, num_steps * dt + dt, dt)    
    plt.subplot(3, 1, 2)
    plt.plot(time, param_estim, label='Param Estimate')
    plt.xlabel('Time')
    plt.ylabel('Parameter Estimate')
    plt.title('Parameter Estimate')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, state_estim[:, 0], label='x1')
    plt.plot(time, state_estim[:, 1], label='x2')
    plt.plot(time, state_estim[:, 2], label='x3')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Estimated state')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()


# Initial conditions
x0 = np.array([0, 1.0, 1.05])
# Control input
u = 0
# Time parameters
dt = 0.01  # time step
num_steps = 10# number of time steps

# Simulate
x, param_estim, state_estim = simulate(x0, u, dt, num_steps)

# Plot
plot_simulation(x, param_estim, state_estim, dt)