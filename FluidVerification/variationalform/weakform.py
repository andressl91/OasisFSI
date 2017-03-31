from fenics import *
from solvers.newtonsolver import Newton_manual

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def mixedformulation(mesh, v_deg, p_deg, T, dt, rho, mu, Um, H, **problem_namespace):

    #Mesh
    x = SpatialCoordinate(mesh)

    #Exact Solution
    inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2)*(1 - cos(pi/2.*t))/2."\
    ,"0"), t = 0, Um = Um, H = H, degree = 3)


    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", v_deg)
    Q = FunctionSpace(mesh, "CG", p_deg)
    VQ = MixedFunctionSpace([V, Q])

    # BOUNDARIES
    Inlet  = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2.5))
    Walls  = AutoSubDomain(lambda x: "on_boundary" and near(x[1], 0) or near(x[1], 0.41))

    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    DomainBoundary().mark(boundaries, 1)
    Inlet.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)
    Walls.mark(boundaries, 4)

    ds = Measure("ds", subdomain_data = boundaries)
    n = FacetNormal(mesh)

    u_inlet = DirichletBC(VQ.sub(0), inlet, boundaries, 2)
    nos_geo = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    p_out = DirichletBC(VQ.sub(1), 0, boundaries, 3)

    bcs = [u_inlet, nos_geo, nos_wall, p_out]

    # Define coefficients
    k = Constant(dt)

    # Create functions
    u_ = {}; p_ = {}; up_ = {}
    for time in ["n", "n-1", "n-2"]:
        up = Function(VQ)
        up_[time] = up
        u, p = split(up_[time])
        u_[time] = u
        p_[time] = p

    v, q = TestFunctions(VQ)

    # Navier-Stokes mixed formulation
    F = rho/k*inner(u_["n"] - u_["n-1"], v)*dx \
        + rho*inner(dot(grad(u_["n"]), u_["n"]), v)*dx \
        + inner(sigma_f(p_["n"], u_["n"], mu), grad(v))*dx \
        + inner(div(u_["n"]), q)*dx

    Lift, Drag, Time = Newton_manual(F, VQ, u_, p_, up_, inlet, bcs, T, dt, n, mu, ds)

    return Lift, Drag, Time, VQ.dim(), mesh.num_cells()
