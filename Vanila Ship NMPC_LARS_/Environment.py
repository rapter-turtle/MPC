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
T = 200.00  # maximum simulation time[s]
dt = 1
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
state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 40, 10, 0])
x0 = x_init
tcomp_sum = 0
tcomp_max = 0
u0 = np.array([0, 0])


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

        
    # current_plot(horizon, ax, round(i*dt,2))    

    state = update_state(state, u0, dt, np.array([0.0,0.0,0]), np.array([2.0,0.0,0]))
    x0 = np.array([state[0], state[1], state[2], state[3] - state[8], state[4]-state[9], state[5], state[6], state[7]])
    # x0 = acados_solver.get(1, "x")

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
plotRes(simX, simU, t)
plotTrackProj(simX)
# current_estim_plot(current, t)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / int(T/dt)))
print("Maximum computation time: {}".format(tcomp_max))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
