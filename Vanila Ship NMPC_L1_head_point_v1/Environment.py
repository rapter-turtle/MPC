import numpy as np
import math 
from USV import *
from USV_model_update import *
import time, os
from simple_acados_settings_dev import *
from simple_plotFcn import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

Tf = 30.0  # prediction horizon
N = 30  # number of discretization steps
T = 50.00  # maximum simulation time[s]
dt = 0.001
control_time = 1

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

x_init = model.x0
state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0])
state_before = state
x0 = x_init
x0_before = x0
tcomp_sum = 0
tcomp_max = 0
u0 = np.array([0, 0])

l = 3.5
m = 3980
Iz = 19703


x_estim = np.array([l - 10 ,0 ,0 ,0 ,0 ,0])
x_error = np.array([0 ,0 ,0 ,0 ,0 ,0])
param_estim = np.array([0.0, 0.0])
filtered_param = np.array([0.0,0.0])
l1_u = np.array([0.0, 0.0])

sim_x_estim_error = np.ndarray((int(T/dt),6))
sim_param = np.ndarray((int(T/dt),2))
sim_l1_con = np.ndarray((int(T/dt),2))
real = np.ndarray((int(T/dt),6))

px = 0.0
py = 0.0
before_px = 0.0
before_py = 0.0


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

    # u0 = np.array([0,0])
    state, V_ship = update_state(state, u0, l1_u, dt, np.array([0.0,0.0,0]), np.array([1.0,0.0,0]),i*dt)
    x0 = np.array([state[0], state[1], state[2], state[3] + l*math.cos(state[5]) - state[8], state[4] + l*math.sin(state[5]) - state[9], state[5], state[6], state[7]])
    # x0 = acados_solver.get(1, "x")

    ############################## L1 Adaptive control ######################################
    Gamma_x = 150
    Gamma_y = 150
    cutoff = 200
    alpha = dt*cutoff

    u_mpc = np.array([state[6], state[7]])
    x_estim =  estim_update_state(x_estim, x_error, u_mpc, dt, np.array([before_px, before_py]), i*dt) #+ np.dot(Am, x_error)  

    uu = state[0]
    v = state[1]
    r = state[2] 
    psi = state[5]
    x_l1 = np.array([state[3] + l*math.cos(state[5]) - state[8],                              ## et1
                     state[4] + l*math.sin(state[5]) - state[9],                              ## et2
                     uu*math.cos(psi) - v*math.sin(psi) - r*l*math.sin(psi) - V_ship[0],      ## et3
                     uu*math.sin(psi) + v*math.cos(psi) + r*l*math.cos(psi) - V_ship[1],      ## et4
                     psi,                                                                     ## psi
                     r])                                                                      ## r
    
    x_error = x_estim - x_l1

    px = Gamma_x*dt*param_dynamics(x_error, before_px, np.array([0, 0, 1, 0, 0, 0]), 100) + before_px
    py = Gamma_y*dt*param_dynamics(x_error, before_py, np.array([0, 0, 0, 1, 0, 0]), 100) + before_py
    

    param_estim[0] = px
    param_estim[1] = py 

    filtered_param = (1-alpha)*filtered_param + alpha*np.array([px, py])

    # l1_u[0] = - (filtered_param[1]*math.sin(psi) + filtered_param[0]*math.cos(psi))*m
    # l1_u[1] = - (filtered_param[1]*math.cos(psi) - filtered_param[0]*math.sin(psi))/(1/m + 4*l/Iz)

    l1_u[0] = - (py*math.sin(psi) + px*math.cos(psi))*m
    l1_u[1] = - (py*math.cos(psi) - px*math.sin(psi))/(1/m + 4*l/Iz)
    # l1_u[0] = - (py*math.sin(psi) + px*math.cos(psi))*m
    # l1_u[1] = - (py*math.cos(psi) - px*math.sin(psi))*(1/m + 4*l/Iz)**(-1)


    sim_l1_con[i,:] = l1_u
    sim_param[i,:] = param_estim
    sim_x_estim_error[i,:] = x_estim
    real[i,:] = x_l1
    

    before_px = px
    before_py = py
    ##########################################################################################



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
plotRes(simX, simU, t, sim_l1_con, sim_param, sim_x_estim_error, real)
plotTrackProj(simX)
# current_estim_plot(current, t)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / int(T/dt)))
print("Maximum computation time: {}".format(tcomp_max))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
