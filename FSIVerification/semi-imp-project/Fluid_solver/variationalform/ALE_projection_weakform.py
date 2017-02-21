from fenics import *
from utils.mapping import *
from utils.stress import *
from solvers.ALE_projection_fluid import ALE_projection_fluid_solver


def ALE_projection_scheme(N, v_deg, p_deg, T, dt, rho, mu, **problem_namespace):

    #Mesh
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    #Exact Solution
    d_e = Expression(("sin(x[1])*sin(t)",
                      "sin(x[0])*sin(t)"
                     ), degree = 4, t = 0)

    w_e = Expression(("cos(x[1])*sin(t)",
                      "sin(x[0])*sin(t)"
                     ), degree = 4, t = 0)

    u_e = Expression(("cos(x[1])*sin(t)",
                      "sin(x[0])*sin(t)"
                     ), degree = 4, t = 0)

    p_e = Expression("sin(x[0])*sin(t)", degree = 4, t = 0)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", v_deg)
    Q = FunctionSpace(mesh, "CG", p_deg)
    VV = V*Q

    # Define coefficients
    k = Constant(dt)

    # Create functions
    u_ = {}; p_ = {}; up_ = {}
    for time in ["n", "n-1", "n-2"]:
        if time == "n":
            tmp_up = Function(VV)
            up_[time] = tmp_up
            up = TrialFunction(VV)
            u, p = split(up)
        else:
            up = Function(VV)
            up_[time] = up
            u, p = split(up)

        u_[time] = u
        p_[time] = p

    v, q = TestFunctions(VV)

    #Functions for solving tentative velocity
    psi = TestFunction(V)
    u_t = TrialFunction(V)
    u_tilde = Function(V)

    #Assigning initial condition
    assign(up_["n-1"].sub(0), project(u_e, V))
    assign(up_["n-1"].sub(1), project(p_e, Q))
    u0_tilde = project(u_e, V)

    #Create symbolic sourceterm
    t_ = Constant(dt)
    d_x = cos(x[1])*sin(t_)
    d_y = cos(x[0])*sin(t_)
    d_vec = as_vector([d_x, d_y])

    w_x = cos(x[1])*cos(t_)
    w_y = cos(x[0])*cos(t_)
    w_vec = as_vector([w_x, w_y])

    u_x = cos(x[1])*cos(t_)
    u_y = cos(x[0])*cos(t_)
    u_vec = as_vector([u_x, u_y])
    p_c = sin(x[0])*sin(t_)

    f = rho*diff(u_vec, t_) + rho*dot(u_vec, grad(u_vec)) - div(sigma_f(p_c, u_vec, mu))

    #Define boundary condition
    bcu = [DirichletBC(V, u_e, "on_boundary")]
    bcs = [DirichletBC(VV.sub(0), u_e, "on_boundary"), DirichletBC(VV.sub(1), p_e, "on_boundary")]

    F1 = (rho/k)*inner(J_(d_vec)*(u_t - u_["n-1"]), psi)*dx
    F1 += rho*inner(J_(d_vec)*inv(F_(d_vec))*dot(u0_tilde - w_vec, grad(u_t)), psi)*dx
    F1 += inner(J_(d_vec)*mu*sigma_f_shearstress_map(u_t, d_vec)*inv(F_(d_vec)).T, sym(grad(psi)))*dx
    F1 -= inner(J_(d_vec)*f, psi)*dx

    # Pressure update
    F2 = rho/k*inner(J_(d_vec)*(u_["n"] - u_tilde), v)*dx \
        - inner(J_(d_vec)*p_["n"], div(v))*dx \
        + inner(div(J_(d_vec)*inv(F_(d_vec))*u_["n"]), q)*dx
        #- inner(p, div(J_(d_vec)*inv(F_(d_vec))*v))*dx \

    #Solve Numerical Problem
    #TODO: Rethink u_tilde
    u_error, p_error, time = ALE_projection_fluid_solver(F1, F2, u_, p_,
                up_, u_tilde, u0_tilde,bcu, bcs, T, dt, u_e, p_e, w_e, d_e, t_)

    return u_error, p_error, mesh.hmin(), time
