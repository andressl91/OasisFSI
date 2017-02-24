from fenics import *
from utils.mapping import *
from utils.stress import *
from solvers.ALE_projection_fluid import ALE_projection_fluid_solver

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def sigma_f_new(u, p, d, mu):
	return -p*Identity(2) + mu*(dot(grad(u), inv(F_(d))) + dot(inv(F_(d)).T, grad(u).T))

def ALE_projection_scheme(N, v_deg, p_deg, T, dt, rho, mu, **problem_namespace):

    #Mesh
    mesh = UnitSquareMesh(N, N)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    #Exact Solution
    d_e = Expression(("0.5*( (x[0]+x[1]+t) - 0.5*(sin(2*(x[0]+x[1]+t))) )",
                      "0.5*( (x[0]+x[1]+t) + 0.5*(sin(2*(x[0]+x[1]+t))) )"
                     ), degree = 4, t = 0)

    w_e = Expression(("pow(sin(x[0] + x[1] + t), 2)",
                      "pow(cos(x[0] + x[1] + t), 2)"
                     ), degree = 5, t = 0)

    u_e = Expression(("pow(sin(x[0] + x[1] + t), 2)",
                      "pow(cos(x[0] + x[1] + t), 2)"
                     ), degree = 5, t = 0)

    p_e = Expression("cos(x[0] + x[1] + t)", degree = 5, t = 0)

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
    d_x = 0.5*( (x[0]+x[1]+t_) - 0.5*(sin(2*(x[0]+x[1]+t_))) )
    d_y = 0.5*( (x[0]+x[1]+t_) + 0.5*(sin(2*(x[0]+x[1]+t_))) )
    d_vec = as_vector([d_x, d_y])

    w_x = sin(x[0] + x[1] + t_)**2
    w_y = cos(x[0] + x[1] + t_)**2
    w_vec = as_vector([w_x, w_y])


    u_x = sin(x[0] + x[1] + t_)**2
    u_y = cos(x[0] + x[1] + t_)**2
    u_vec = as_vector([u_x, u_y])
    p_c = cos(x[0] + x[1] + t_)


    #Define boundary condition
    #w_e due to u_tilde = w on solid interface
    bcu = [DirichletBC(V, w_e, "on_boundary")]
    bcs = [DirichletBC(VV.sub(0), u_e, "on_boundary"), DirichletBC(VV.sub(1), p_e, "on_boundary")]

    f = rho*diff(u_vec, t_) + rho*dot(u_vec - w_vec, grad(u_vec)) - div(sigma_f(p_c, u_vec, mu))
    f_map = J_(d_vec)*rho*diff(u_vec, t_) \
    + J_(d_vec)*rho*inv(F_(d_vec))*dot((u_vec - w_vec), grad(u_vec))\
    - div(J_(d_vec)*sigma_f_new(u_vec, p_c, d_vec, mu)*inv(F_(d_vec)).T)


    F1 = (rho/k)*inner(J_(d_vec)*(u_t - u_["n-1"]), psi)*dx
    F1 += rho*inner(J_(d_vec)*inv(F_(d_vec))*dot(u0_tilde - w_vec, grad(u_t)), psi)*dx
    F1 += inner(J_(d_vec)*mu*sigma_f_shearstress_map(u_t, d_vec)*inv(F_(d_vec)).T, sym(grad(psi)))*dx
    F1 -= inner(f_map, psi)*dx
    #F1 -= inner(J_(d_vec)*f, psi)*dx

    # Pressure update
    F2 = rho/k*inner(J_(d_vec)*(u_["n"] - u_tilde), v)*dx \
    - inner(J_(d_vec)*p_["n"], div(v))*dx \
    - inner(div(J_(d_vec)*inv(F_(d_vec))*u_["n"]), q)*dx \
    + J_(d_vec)*inner(dot(u_["n"], n) - dot(w_vec, n), dot(v, n))*ds
        #- inner(p_["n"], div(J_(d_vec)*inv(F_(d_vec))*v))*dx \

    #Solve Numerical Problem
    #TODO: Rethink u_tilde
    u_error, p_error, time = ALE_projection_fluid_solver(F1, F2, u_, p_,
                up_, u_tilde, u0_tilde,bcu, bcs, T, dt, u_e, p_e, w_e, d_e, t_)

    return u_error, p_error, mesh.hmin(), time
