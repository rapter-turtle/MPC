import numpy as np
import math 
from USV import *
from USV_model_update import *
import time, os
from simple_acados_settings_dev import *
from simple_plotFcn import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

Tf = 50.0  # prediction horizon
N = 50  # number of discretization steps
T = 100  # maximum simulation time[s]
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

x_pos = 10.0
y_pos = 0.0

x_init = model.x0
state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, y_pos, 0])
nominal_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, y_pos, 0])
state_before = state
x0 = x_init
x0_before = x0
tcomp_sum = 0
tcomp_max = 0
u0 = np.array([0.0, 0.0])

l = 3.5
m = 3980.0
Iz = 19703.0


x_estim = np.array([l - 10.0 ,-y_pos ,0.0 ,0.0 ,0.0 ,0.0])
x_estim_before = np.array([l - 10.0 ,-y_pos ,0.0 ,0.0 ,0.0 ,0.0])
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


    l1_u = np.array([0,0])
    nominal_state, V_ship = update_state(nominal_state, u0, l1_u, dt, np.array([0.0,0.0,0]), np.array([1.0,0.0,0]),i*dt)

    state = nominal_state + np.array([0, 0.25*math.cos(0.5*i*dt), 0, 0, 0.5*math.sin(0.5*i*dt), 0, 0, 0, 0, 0, 0])
    
    x0 = np.array([state[0], state[1], state[2], state[3] + l*math.cos(state[5]) - state[8], state[4] + l*math.sin(state[5]) - state[9], state[5], state[6], state[7]])

    ############################## L1 Adaptive control ######################################
    Gamma_x = 10000
    Gamma_y = 100000
    Gamma_psi = 10000
    w_cutoff = 50

    u_mpc = np.array([state[6], state[7]])
    x_estim =  estim_update_state(x_estim_before, x_error, u_mpc, filtered_param, dt, np.array([px, py, ppsi]), i*dt) #+ np.dot(Am, x_error)  

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

    px = Gamma_x*dt*param_dynamics(x_error, before_px, np.array([0, 0, 1, 0, 0, 0]), 0.7) + before_px
    py = Gamma_y*dt*param_dynamics(x_error, before_py, np.array([0, 0, 0, 1, 0, 0]), 0.7) + before_py
    ppsi = Gamma_psi*dt*param_dynamics(x_error, before_ppsi, np.array([0, 0, 0, 0, 0, 1]), 1000) + before_ppsi
          

    param_estim[0] = px
    param_estim[1] = py
    param_estim[2] = ppsi

    #################################### DOB #################################### 

    for k in range(N+1):
        acados_solver.set(k,"p", np.array([px, py, ppsi]))

    print(px)

    #################################### DOB ####################################


    sim_l1_con[i,:] = l1_u
    sim_param[i,:] = param_estim
    sim_x_estim_error[i,:] = x_estim
    real[i,:] = x_l1

    before_px = px
    before_py = py
    before_ppsi = ppsi
    x_estim_before = x_estim

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
