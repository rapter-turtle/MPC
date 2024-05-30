import numpy as np
import math 

def update_state(x_t,bef_con, u, l1_u, dt, V, V_t, t):


    m = 3980
    Iz = 19703
    xdot = 0
    Yvdot = 0
    Yrdot = 0
    Nvdot = 0
    Nrdot = 0
    Xu = -50
    Yv = -200
    Yr = 0
    Nv = 0
    Nr = -1281
    Xuu = -135
    Yvv = -2000
    Nrrr = -3224
    lr = 4
    m11 = m - xdot
    m22 = m - Yvdot 
    m23 = -Yrdot
    m32 = -Nvdot
    m33 = Iz - Nrdot


    uu = x_t[0]
    v = x_t[1]
    r = x_t[2]
    psi = x_t[5]
    Tau_x = x_t[6]
    Tau_y = x_t[7]  



    M = np.array([[m11, 0, 0], [0, m22, m23], [0, m32, m33]])
    
    Cv = np.array([[0, 0, -m22*v-m23*r], [0, 0, m11*uu], [m22*v+m23*r, -m11*uu, 0]])
    
    D = -np.array([[Xu, 0, 0], [0, Yv, Yr], [0, Nv, Nr]])
    
    R = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])

    uvr = np.array([uu, v, r])


    entire = 0.05*math.sin(0.75*t)*m + 0.01*math.sin(3*t)*m + 0.001*math.sin(10*t)*m - 30.0
    disturbance_x = entire*math.sin(50*math.pi/180)
    disturbance_y = entire*math.cos(50*math.pi/180)

    con_Tau_raw = np.array([Tau_x + l1_u[0], Tau_y + l1_u[1] , -4*(Tau_y + l1_u[1])]) + np.array([u[0], u[1], -4*u[1]])*dt


    tx = 0.0
    ty = 0.0

    # print("con : ",u[0]*dt)
    # print("l1 : ",l1_u[0])

    # if (con_Tau_raw[0] - bef_con[0]) >= 0.2:
    #     tx = 0.2
    # if (con_Tau_raw[0] - bef_con[0]) <= -0.2:
    #     tx = -0.2

    # if (con_Tau_raw[1] - bef_con[1]) >= 0.05:
    #     ty = 0.05
    # if (con_Tau_raw[1] - bef_con[1]) <= -0.05:
    #     ty = -0.05

    # con_Tau = np.array([Tau_x + tx, Tau_y + ty , -4*(Tau_y + ty)]) 


    con_Tau = np.array([Tau_x + l1_u[0], Tau_y + l1_u[1] , -4*(Tau_y + l1_u[1])]) + np.array([u[0], u[1], -4*u[1]])*dt
    

    if con_Tau[0] > 799.999:
        con_Tau[0] = 799.99
    if con_Tau[0] < -799.999:
        con_Tau[0] = -799.99

    if con_Tau[1] > 199.999:
        con_Tau[1] = 199.99
    if con_Tau[1] < -199.999:
        con_Tau[1] = -199.99


    Tau = np.array([con_Tau[0] + disturbance_x, con_Tau[1] + disturbance_y, -4*(con_Tau[1])])
    # Tau = Tau + np.array([u[0], u[1], -4*u[1]])*dt

    

    M_inv = np.linalg.inv(M)

    uvr_dot_bM = (Tau - np.dot(Cv, uvr) - np.dot(D, uvr))
    uvr_dot = np.dot(M_inv, uvr_dot_bM)
    xypsi_dot = np.dot(R, uvr) 


    V_y = V_t[1]


    # V_y = 0.75*math.sin(0.75*t)
    V_t[1] = V_y

    xdot = np.concatenate((uvr_dot, xypsi_dot, u, V_t), axis=0)
    

    x_t_plus_1 = xdot * dt + x_t

    real_psi = x_t_plus_1[5]
    if real_psi > math.pi:
        x_t_plus_1[5] -= 2*math.pi
    if real_psi < -math.pi:
        x_t_plus_1[5] += 2*math.pi
        
    
    # if x_t_plus_1[6] > 799.999:
    #     x_t_plus_1[6] = 799.99
    # if x_t_plus_1[6] < -799.999:
    #     x_t_plus_1[6] = -799.99

    # if x_t_plus_1[7] > 199.999:
    #     x_t_plus_1[7] = 199.99
    # if x_t_plus_1[7] < -199.999:
    #     x_t_plus_1[7] = -199.99



    V_ship = np.array([1.0, V_y, 0.0])

    return x_t_plus_1, V_ship, con_Tau