import numpy as np
import math 
from USV import *
from USV_model_update import *
import time, os
from simple_acados_settings_dev import *
from simple_plotFcn import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

Tf = 25.0  # prediction horizon
N = 50  # number of discretization steps
T = 150  # maximum simulation time[s]
dt = 0.001
control_time = 0.5



# load model
constraint, model, acados_solver = acados_settings(Tf, N)

# dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.ndarray((int(T/dt), nx+3))
simU = np.ndarray((int(T/dt), nu))

x_pos = 4.0
y_pos =2.0

x_init = model.x0
state = np.array([0, 0, 0, 0, 0, 0, 0, 0, x_pos, y_pos, 0])
state_before = state
x0 = x_init
x0_before = x0
tcomp_sum = 0
tcomp_max = 0
u0 = np.array([0.0, 0.0])

l = 3.5
m = 3980.0
Iz = 19703.0


x_estim = np.array([l - x_pos ,-y_pos ,0.0 ,0.0 ,0.0 ,0.0])
x_estim_before = np.array([l - x_pos ,-y_pos ,0.0 ,0.0 ,0.0 ,0.0])
x_error = np.array([0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0])
param_estim = np.array([0.0, 0.0, 0.0])
filtered_param = np.array([0.0,0.0, 0.0])
l1_u = np.array([0.0, 0.0])

sim_x_estim_error = np.ndarray((int(T/dt),6))
sim_param = np.ndarray((int(T/dt),3))
sim_l1_con = np.ndarray((int(T/dt),2))
real = np.ndarray((int(T/dt),6))
sim_filtered = np.ndarray((int(T/dt),3))

px = 0.0
py = 0.0
ppsi = 0.0
before_px = 0.0
before_py = 0.0
before_ppsi = 0.0

# simulate
for i in range(int(T/dt)):
    if i%int(control_time/dt) == 0:
        # print(i)
        ################################ update reference ################################
        for j in range(N):
            yref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            acados_solver.set(j, "yref", yref)
        yref_N = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        acados_solver.set(N, "yref", yref_N)
        ##################################################################################

        # solve ocp
        t = time.time()

        status = acados_solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))

        elapsed = time.time() - t
        
        # manage timings
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # get solution
        u0 = acados_solver.get(0, "u")
        

        horizon = []
        for k in range(N):
            horizon.append(acados_solver.get(k, "x"))

        current_plot(horizon, ax, round(i*dt,2), state)     



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

        n2 = X_train.shape[0]
        K = np.zeros(n2)
        X_new = np.array([state[3] + l*math.cos(state[5]) - state[8],state[4] + l*math.sin(state[5]) - state[9], state[5], np.sqrt(state[0]**2+state[1]**2), state[2]])

        for i in range(n2):
            diff = X_new - X_train[i, :]
            sqdist = np.dot(diff, diff)
            K[i] = variance * np.exp(-0.5 * sqdist / length_scale**2)
                
        mu = np.dot(K,K_train)
        print(mu)




    # l1_u = np.array([0,0])
    state, V_ship = update_state(state, u0, dt, np.array([0.0,0.0,0]), np.array([1.0,0.0,0]),i*dt)

    
    x0 = np.array([state[0], state[1], state[2], -(state[3] + l*math.cos(state[5]) - state[8]), -(state[4] + l*math.sin(state[5]) - state[9]), state[5], state[6], state[7]])



    for j in range(nx + 3):
        simX[i, j] = state[j]
    for j in range(nu):
        simU[i, j] = u0[j]


    # update initial condition
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    # acados_solver.set("x0", x)
 
    

# Plot Results
t = np.linspace(0.0, T, int(T/dt))
plotRes(simX, simU, t, sim_l1_con, sim_param, sim_x_estim_error, real, sim_filtered)
plotTrackProj(sim_x_estim_error, simX, int(T/dt),t)
# current_estim_plot(current, t)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / int(T/dt)))
print("Maximum computation time: {}".format(tcomp_max))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
