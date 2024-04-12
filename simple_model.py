

from casadi import *
import numpy as np
import math


def simple_model():
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "simple_model"

    ## CasADi Model
    # set up states & controls

    uu = MX.sym("uu")
    v = MX.sym("v")
    r = MX.sym("r")
    xx = MX.sym("xx")
    y = MX.sym("y")
    psi = MX.sym("psi")
    Tau_x = MX.sym("Tau_x")
    Tau_y = MX.sym("Tau_y")
    Tau_psi = MX.sym("Tau_psi")
    Tau_x_dot = MX.sym("Tau_x_dot")
    Tau_y_dot = MX.sym("Tau_y_dot")
    Tau_psi_dot = MX.sym("Tau_psi_dot")

    uvr = vertcat(uu, v, r)
    # Tau_dot = np.array([Tau_x_dot],[Tau_y_dot],[Tau_psi_dot])

    # controls
    Tau_x_dot_control = MX.sym("Tau_x_dot_control")
    Tau_y_dot_control = MX.sym("Tau_y_dot_control")
    Tau_psi_dot_control = MX.sym("Tau_psi_dot_control")
    

    u = vertcat(Tau_x_dot_control, Tau_y_dot_control, Tau_psi_dot_control)

    # xdot
    u_dot = MX.sym("u_dot")
    v_dot = MX.sym("v_dot")
    r_dot = MX.sym("r_dot")
    x_dot = MX.sym("x_dot")
    y_dot = MX.sym("y_dot")
    psi_dot = MX.sym("psi_dot")
    xdot = vertcat(u_dot,v_dot,r_dot,x_dot,y_dot,psi_dot,Tau_x_dot,Tau_y_dot,Tau_psi_dot)
    x = vertcat(uu, v, r, xx, y, psi, Tau_x, Tau_y, Tau_psi)
    Tau = vertcat(Tau_x, Tau_y, Tau_psi)

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])

    # constant
    m11 = 4.5096*1000000
    m22 = 7.5608*1000000
    m23 = -2.2680*1000000
    m32 = -2.2680*1000000
    m33 = 2.9683*100000000
    Xu = -5.1380*10000
    Yv = -1.6980*10000
    Yr = 1.5081*1000000
    Nv = 1.5081*1000000
    Nr = -2.53*100000000

    M = vertcat(
        horzcat(m11, 0, 0),
        horzcat(0, m22, m23),
        horzcat(0, m32, m33)
    )

    Cv = vertcat(
        horzcat(0, 0, -m22*v-m23*r),
        horzcat(0, 0, m11*uu),
        horzcat(m22*v+m23*r, -m11*uu, 0)
    )

    D = -vertcat(
        horzcat(Xu, 0, 0),
        horzcat(0, Yv, Yr),
        horzcat(0, Nv, Nr)
    )
    

    R = vertcat(
        horzcat(cos(psi), -sin(psi), 0),
        horzcat(sin(psi), cos(psi), 0),
        horzcat(0, 0, 1)
    )

    uvr_dot_bM = (Tau - Cv@uvr - D@uvr)
    uvr_dot = inv(M)@uvr_dot_bM

    xypsi_dot = R@uvr
    # dynamics
#######################################################################################################################
    

#######################################################################################################################

    f_expl = vertcat(
        uvr_dot,
        xypsi_dot,
        u,
    )

    # Model bounds
    model.Tau_dot_x_min = -1000  
    model.Tau_dot_x_max = 1000  
    model.Tau_dot_y_min = -100  
    model.Tau_dot_y_max = 100
    model.Tau_dot_psi_min = -100  
    model.Tau_dot_psi_max = 100

    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    constraint.expr = vertcat([])


    # Define model struct
    params = types.SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint

