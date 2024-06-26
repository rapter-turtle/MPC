import numpy as np
import math 

def update_state(x_t, u, dt, V):


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
    m11 = m 
    m22 = m 
    m23 = -Yrdot
    m32 = -Nvdot
    m33 = Iz - Nrdot


    xxdot = x_t[0]
    yydot = x_t[1]
    r = x_t[2]
    psi = x_t[5]
    Tau_x = x_t[6]
    Tau_y = x_t[7] 
    uu = (xxdot - V[0])*math.cos(psi) + (yydot - V[1])*math.sin(psi)
    v = -(xxdot - V[0])*math.sin(psi) + (yydot - V[1])*math.cos(psi)

    M = np.array([[m11, 0, 0], [0, m22, m23], [0, m32, m33]])
    
    Cv = np.array([[0, 0, -m22*v-m23*r], [0, 0, m11*uu], [m22*v+m23*r, -m11*uu, 0]])
    
    D = -np.array([[Xu, 0, 0], [0, Yv, Yr], [0, Nv, Nr]])
      
    R = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])

    R_dot = np.array([[-r*math.sin(psi), -r*math.cos(psi), 0],[r*math.cos(psi), -r*math.sin(psi), 0],[0, 0, 0]])
    

    uvr = np.transpose(np.array([uu, v, r]))

    if Tau_x >= 800:
        Tau_x = 799.999999999
    if Tau_x <= -800:
        Tau_x = -799.999999999
    if Tau_y >= 200:
        Tau_y = 199.999999999
    if Tau_x <= -200:
        Tau_x = -199.999999999                        

    Tau = np.transpose(np.array([Tau_x, Tau_y, 4*Tau_y])) 


    M_inv = np.linalg.inv(M)

    uvr_dot_bM = Tau - np.dot(D, uvr) - np.dot(Cv, uvr) 
    uvr_dot = np.dot(M_inv, uvr_dot_bM)
    xypsi_dot = np.dot(R, uvr) + np.transpose(V) 
    xypsi_dotdot = np.dot(R_dot, uvr) + np.dot(R, uvr_dot)

    x_dot = np.concatenate((xypsi_dotdot, xypsi_dot, u), axis=0)


    a = x_dot * dt + x_t
    # print("   ")
    # print("Current_state : " ,x_t)
    # print("Control : " ,u)
    # print("Next_state : " , a)

    # x_t_plus_1 = np.concatenate((a,[a[0]*math.cos(a[5]) - a[1]*math.sin(a[5])]), axis=0) 
    # print(x_t_plus_1)

    return a

