from fenics import *
from solvers.projection_fluid import projection_fluid_solver

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def semi_projection_scheme(N, v_deg, p_deg, T, dt, rho, mu, **problem_namespace):

    #Mesh
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    #Exact Solution
    u_e = Expression(("pow(sin(x[0] + x[1] + t), 2)",
                      "pow(cos(x[0] + x[1] + t), 2)"
                     ), degree = 5, t = 0)

    p_e = Expression("pow(cos(x[0] + x[1] + t), 2)", degree = 5, t = 0)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", v_deg)
    Q = FunctionSpace(mesh, "CG", p_deg)
    VV = MixedFunctionSpace([V, Q])

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
    u_x = sin(x[0] + x[1] + t_)**2
    u_y = cos(x[0] + x[1] + t_)**2
    u_vec = as_vector([u_x, u_y])
    p_c = cos(x[0] + x[1] + t_)**2

    f = rho*diff(u_vec, t_) + rho*dot(u_vec, grad(u_vec)) - div(sigma_f(p_c, u_vec, mu))

    #Define boundary condition
    bcu = [DirichletBC(V, u_e, "on_boundary")]
    bcs = [DirichletBC(VV.sub(0), u_e, "on_boundary"), DirichletBC(VV.sub(1), p_e, "on_boundary")]

    # Advection-diffusion step (explicit coupling)
    F1 = (rho/k)*inner(u_t - u_["n-1"], psi)*dx + rho*inner(dot(u0_tilde, grad(u_t)), psi)*dx \
         + 2.*mu*inner(sym(grad(u_t)), sym(grad(psi)))*dx - inner(f, psi)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    F2 = rho/k*inner(u_["n"] - u_tilde, v)*dx - inner(p_["n"], div(v))*dx \
    - inner(q, div(u_["n"]))*dx

    #Solve Numerical Problem
    #TODO: Rethink u_tilde
    u_error, p_error, time = projection_fluid_solver(F1, F2, u_, p_, up_, u_tilde, u0_tilde,bcu, bcs, T, dt, u_e, p_e, t_)

    return u_error, p_error, mesh.hmin(), time
