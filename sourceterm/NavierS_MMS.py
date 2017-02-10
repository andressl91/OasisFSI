from fenics import *
import numpy as np

def eps(U):
    return 0.5*(grad(U) + grad(U).T)

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def solver(N, dt, T):
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
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
    p_e = Expression("2*sin(x[0])*cos(x[1])*sin(t)", t = dt)
    t = Constant(dt)

    #t_eval = dt
    u_x = x[1]*cos(x[1])*sin(t)
    u_y = -x[0]*sin(x[0])*sin(t)
    p_c = 2*sin(x[0])*cos(x[1])*sin(t)
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
        t.assign(t_step)

        J = derivative(F, up, phi)
        solve(F == 0, up, bcs, J = J)
        up0.assign(up)
        t_step += dt

    u_, p_ = up.split(True)
    u_e.t = t_step - dt
    p_e.t = t_step - dt
    L2_u = errornorm(u_e, u_, norm_type="l2", degree_rise = 2)
    L2_p = errornorm(p_e, p_, norm_type="l2", degree_rise = 2)
    E_u.append(L2_u)
    E_p.append(L2_p)
    h.append(mesh.hmin())


N = [4, 8, 12, 16]
dt = [0.00005]
T = 0.005
E_u = [];  E_p = []; h = []
for n in N:
    for t in dt:
        print "Solving for t = %g, N = %d" % (t, n)
        solver(n, t, T)

print "Checking Convergence in Space P2-P1"

for i in E_u:
    print "Errornorm Velocity L2", i

print

for i in range(len(E_u) - 1):
    r_u = np.log(E_u[i+1]/E_u[i]) / np.log(h[i+1]/h[i])
    print "Convergence Velocity", r_u

print

for i in E_p:
    print "Errornorm Pressure L2", i

print

for i in range(len(E_p) - 1):
    r_p = np.log(E_p[i+1]/E_p[i]) / np.log(h[i+1]/h[i])

    print "Convergence Pressure", r_u

print "Checking Convergence in time"

N = [64]
dt = [0.05, 0.04, 0.02, 0.01]
T = 0.4
E_u = [];  E_p = []; h = []
for n in N:
    for t in dt:
        print "Solving for t = %g, N = %d" % (t, n)
        solver(n, t, T)

for i in E_u:
    print "Errornorm Velocity L2", i

print

for i in range(len(E_u) - 1):
    r_u = np.log(E_u[i+1]/E_u[i]) / np.log(dt[i+1]/dt[i])
    print "Convergence Velocity", r_u

print

for i in E_p:
    print "Errornorm Pressure L2", i

print

for i in range(len(E_p) - 1):
    r_p = np.log(E_p[i+1]/E_p[i]) / np.log(dt[i+1]/dt[i])

    print "Convergence Pressure", r_u
