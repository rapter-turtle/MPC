import numpy as np
import math 
from USV import *
import time, os
from simple_acados_settings_dev import *
from simple_plotFcn import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

Tf = 100.0  # prediction horizon
N = 100  # number of discretization steps
T = 500.00  # maximum simulation time[s]
dt = 0.01
control_time = 1

# load model
constraint, model, acados_solver = acados_settings(Tf, N)

V_current = np.array([-2,3,0])

# dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.ndarray((int(T/dt), nx))
simU = np.ndarray((int(T/dt), nu))

x_init = np.array([V_current[0], V_current[1], 0, 0, 0, 0, 0, 0])
x0 = x_init
tcomp_sum = 0
tcomp_max = 0
u0 = np.array([0, 0])

#######################State Observer##########################
k_x1 = 4
k_x2 = 4
k_y1 = 4
k_y2 = 4

xydot_init = np.array([0,0,0,0])
xydot = np.array([0,0,0,0])
x_estim = 0
y_estim = 0
Vx_estim = 0
Vy_estim = 0


current = np.ndarray((int(T/dt), 2))
###############################################################


# simulate
for i in range(int(T/dt)):
    if i%int(control_time/dt) == 0:
        # print(i)
        ################################ update reference ################################
        for j in range(N):
            yref = np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            acados_solver.set(j, "yref", yref)
        yref_N = np.array([2, 0, 0, 0, 0, 0, 0, 0])
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
        # current_plot(horizon, ax, round(i*dt,2))

    x0 = update_state(x0, u0, dt, V_current)
    # x0 = acados_solver.get(1, "x")


    #######################State Observer##########################
    Vx_estim = Vx_estim + k_x2*(x0[3] - x_estim)*dt
    # print("1 : ", k_x2*(x0[3] - x_estim)*dt)
    # print("Vx_estim : ", Vx_estim)
    Vy_estim = Vy_estim + k_y2*(x0[4] - y_estim)*dt
    x_estim = x_estim + (x0[0] + k_x1*(x0[3] - x_estim))*dt
    y_estim = y_estim + (x0[1] + k_y1*(x0[4] - y_estim))*dt

    for k in range(N+1):
        acados_solver.set(k,"p", np.array([Vx_estim/dt, Vy_estim/dt]))
        

    current[i, 0] = Vx_estim/dt
    current[i, 1] = Vy_estim/dt

    # print("Vx : ",Vx_estim, ", Vy : ", Vy_estim)
    ###############################################################




    for j in range(nx):
        simX[i, j] = x0[j]
    for j in range(nu):
        simU[i, j] = u0[j]


    # update initial condition
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    # acados_solver.set("x0", x)

    

# Plot Results
t = np.linspace(0.0, T, int(T/dt))
plotRes(simX, simU, t)
plotTrackProj(simX, t)
current_estim_plot(current, t)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / int(T/dt)))
print("Maximum computation time: {}".format(tcomp_max))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
