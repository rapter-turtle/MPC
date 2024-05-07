

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


    xx = MX.sym("xx")
    y = MX.sym("y")
    psi = MX.sym("psi")
    r = MX.sym("r")    
    Tau_x = MX.sym("Tau_x")
    Tau_y = MX.sym("Tau_y")
    Tau_x_dot = MX.sym("Tau_x_dot")
    Tau_y_dot = MX.sym("Tau_y_dot")
    x_dotdot = MX.sym("x_dotdot")
    y_dotdot = MX.sym("y_dotdot")


    # Tau_dot = np.array([Tau_x_dot],[Tau_y_dot],[Tau_psi_dot])

    # controls
    Tau_x_dot_control = MX.sym("Tau_x_dot_control")
    Tau_y_dot_control = MX.sym("Tau_y_dot_control")

    

    u = vertcat(Tau_x_dot_control, Tau_y_dot_control)

    # xdot
    r_dot = MX.sym("r_dot")
    x_dot = MX.sym("x_dot")
    y_dot = MX.sym("y_dot")

    xdot = vertcat(x_dotdot, y_dotdot, r_dot, x_dot, y_dot, r, Tau_x_dot, Tau_y_dot)
    x = vertcat(x_dot, y_dot, r, xx, y, psi, Tau_x, Tau_y)


    Tau = vertcat(Tau_x, Tau_y, 4*Tau_y)

    Vx = MX.sym("Vx")
    Vy = MX.sym("Vy")

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat(Vx, Vy)

    # constant
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
    m11 = m - xdot
    m22 = m - Yvdot 
    m23 = -Yrdot
    m32 = -Nvdot
    m33 = Iz - Nrdot

    uu = (x_dot - Vx)*cos(psi) + (y_dot - Vy)*sin(psi)
    v = -(x_dot - Vx)*sin(psi) + (y_dot - Vy)*cos(psi)

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
    R_dot = vertcat(
        horzcat(-r*sin(psi), -r*cos(psi), 0),
        horzcat(r*cos(psi), -r*sin(psi), 0),
        horzcat(0, 0, 0)
    )
    Current = vertcat(Vx, Vy, 0)

    uvr = vertcat(uu, v, r)
    uvr_dot_bM = Tau - D@uvr - Cv@uvr
    uvr_dot = inv(M)@uvr_dot_bM
    xypsi_dot = R@uvr
    
    # dynamics
    f_expl = vertcat(
        R_dot@uvr + R@uvr_dot,
        xypsi_dot + Current,
        u
    )

    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    

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

