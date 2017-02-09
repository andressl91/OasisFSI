from fenics import *
import numpy as np

def eps(U):
    return 0.5*(grad(U) + grad(U).T)

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def solver(N, dt, T):
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 3)
    Q = FunctionSpace(mesh, "CG", 2)
    W = V*Q

    up = Function(W)
    u, p = split(up)

    phi = TrialFunction(W)
    vq = TestFunction(W)
    v, q = split(vq)

    up0 = Function(W)
    u0, p0 = split(up0)

    k = Constant(dt)
    t_step = dt

    mu = 10
    rho = 1

    u_e = Expression(("x[1]*cos(x[1])*sin(t)",
                    "-x[0]*sin(x[0])*sin(t)"
                     ), t = dt)
    p_e = Expression("2")
    t = variable(dt)
    #t_eval = dt
    u_x = x[1]*cos(x[1])*sin(t)
    u_y = -x[0]*sin(x[0])*sin(t)
    p_c = 2
    u_vec = as_vector([u_x, u_y])

    # Create right hand side f
    f = rho*diff(u_vec, t) + rho*grad(u_vec)*u_vec - div(sigma_f(p_c, u_vec, mu))

    F = rho/k*inner(u - u0, v)*dx + rho*inner(grad(u)*u, v)*dx \
       + inner(sigma_f(p, u, mu), eps(v))*dx \
       - inner(f, v)*dx + inner(div(u), q)*dx

    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)
    bcs = [DirichletBC(W.sub(0), u_e, "on_boundary"), \
           DirichletBC(W.sub(1), p_e, "on_boundary")]

    while t_step <= T:
        u_e.t = t_step
        p_e.t = t_step
        t = variable(t_step)
        #t_eval = t_step
        """
        u_x = x[1]*cos(x[1])*sin(t)
        u_y = -x[0]*sin(x[0])*sin(t)
        p_c = 2
        u_vec = as_vector([u_x, u_y])
        f = rho*diff(u_vec, t) + rho*grad(u_vec)*u_vec - div(sigma_f(p_c, u_vec, mu))
        F = rho/k*inner(u - u0, v)*dx + rho*inner(grad(u)*u, v)*dx \
           + inner(sigma_f(p, u, mu), eps(v))*dx \
           - inner(f, v)*dx + inner(div(u), q)*dx
        """
        J = derivative(F, up, phi)
        solve(F == 0, up, bcs, J = J)
        up0.assign(up)
        t_step += dt

    u_, p_ = up.split(True)
    u_e.t = t_step - dt
    L2 = errornorm(u_e, u_, norm_type="l2", degree_rise = 3)
    E.append(L2)
    h.append(mesh.hmin())


N = [30]
dt = [0.005]

T = 0.01
E = []; h = []
for n in N:
    for t in dt:
        print "Solving for t = %g, N = %d" % (t, n)
        solver(n, t, T)

for i in E:
    print "Errornorm", i

for i in range(len(E) - 1):
    #r = np.log(E[i+1]/E[i]) / np.log(dt[i+1]/dt[i])
    r = np.log(E[i+1]/E[i]) / np.log(h[i+1]/h[i])
    print "Convergence", r
