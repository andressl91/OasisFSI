from fenics import *
from solvers.projection_fluid import projection_fluid_solver

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def semi_projection_scheme(N, u_x, u_y, p_c, v_deg, p_deg, T, dt, rho, mu, **problem_namespace):

    #Mesh
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", v_deg)
    Q = FunctionSpace(mesh, "CG", p_deg)

    # Define coefficients
    k = Constant(dt)
    nu = Constant(mu/rho)

    # Create functions
    u_ = {}; p_ = {};
    for time in ["n", "n-1", "n-2"]:
        if time == "n":
            u = TrialFunction(V)
            p = TrialFunction(Q)
        else:
            u = Function(V)
            p = Function(Q)

        u_[time] = u
        p_[time] = p

    v = TestFunction(V)
    q = TestFunction(Q)
    u_sol = Function(V)
    p_sol = Function(Q)
    #Functions for solving tentative velocity
    psi = TestFunction(V)
    u_t = TrialFunction(V)
    u_tilde = Function(V)

    #Exact Solution
    u_e = Expression((u_x,
                      u_y,
                     ), t_ = 0)

    p_e = Expression(p_c, t_ = 0)

    t_ = Constant(dt)
    exec("u_x = %s" % u_x)
    exec("u_y = %s" % u_y)
    exec("p_c = %s" % p_c)
    u_vec = as_vector([u_x, u_y])

    #Assigning initial condition
    u_["n-1"].assign(interpolate(u_e, V))
    p_["n-1"].assign(interpolate(p_e, Q))
    u0_tilde = interpolate(u_e, V)

    #Create symbolic sourceterm
    f = diff(u_vec, t_) + dot(u_vec, grad(u_vec)) \
    - 1./rho*div(sigma_f(p_c, u_vec, mu))

    f2 = div(u_e)

    #Define boundary condition
    bcu = [DirichletBC(V, u_e, "on_boundary")]
    bcp = [DirichletBC(Q, p_e, "on_boundary")]


    # Tentative velocity step
    F1 = (1./k)*inner(u_["n"] - u_["n-1"], v)*dx + inner(dot(u_["n-1"], grad(u_["n-1"])), v)*dx + \
         nu*inner(grad(u_["n"]), grad(v))*dx - inner(f, v)*dx
         #2*nu*inner(sym(grad(u_["n"])), sym(grad(v)))*dx - inner(f, v)*dx

    # Pressure update
    F2 = inner(grad(p_["n"]), grad(q))*dx + (1./k)*div(u_sol)*q*dx #+ f2*q*dx

    # Velocity update
    F3 = inner(u_["n"], v)*dx - inner(u_sol, v)*dx + k*inner(grad(p_sol), v)*dx

    #Solve Numerical Problem
    #TODO: Rethink u_tilde
    #u_error, p_error, time = projection_fluid_solver(F1, F2, u_, p_, up_, u_tilde, u0_tilde, bcu, bcs, T, dt, u_e, p_e, t_)
    u_error, p_error, time = projection_fluid_solver(**vars())
    return u_error, p_error, mesh.hmin(), time
