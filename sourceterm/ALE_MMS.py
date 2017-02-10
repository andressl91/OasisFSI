from fenics import *
import numpy as np
set_log_active(False)
def eps(U):
    return 0.5*(grad(U) + grad(U).T)

def sigma_f(p_, u_, mu_f):
    return -p_*Identity(2) + mu_f*(grad(u_) + grad(u_).T)

def sigma_f_new(u,p,d,mu_f):
	return -p*Identity(2) + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def F_(U):
	return (Identity(2) + grad(U))

def J_(U):
	return det(F_(U))

def solver(N, dt, T):
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    up = Function(W)
    u, p = split(up)

    #phi = TrialFunction(W)
    phi,gamma = TestFunctions(W)


    up0 = Function(W)
    u0, p0 = split(up0)
    w = Function(V)
    d = Function(V)

    k = Constant(dt)
    t_step = dt

    mu_f = 10
    rho_f = 1

    u_e = Expression(("x[1]*cos(x[1])*sin(t)","-x[0]*sin(x[0])*sin(t)"), t = dt)
    p_e = Expression("2*sin(x[0])*cos(x[1])*sin(t)", t = dt)
    w_e = Expression(("x[1]*cos(x[1])*sin(t)","-x[0]*sin(x[0])*sin(t)"), t = dt)
    d_e = Expression(("-x[1]*cos(x[1])*cos(t)","x[0]*sin(x[0])*cos(t)"), t = dt)

    w = interpolate(w_e,V)
    d = interpolate(d_e,V)
    t = Constant(dt)

    #t_eval = dt
    u_x = x[1]*cos(x[1])*sin(t)
    u_y = -x[0]*sin(x[0])*sin(t)
    p_c = 2*sin(x[0])*cos(x[1])*sin(t)
    d_x = -x[1]*cos(x[1])*cos(t)
    d_y = x[0]*sin(x[0])*cos(t)

    u_vec = as_vector([u_x, u_y])
    w_vec = as_vector([u_x, u_y])
    d_vec = as_vector([d_x, d_y])


    # Create right hand side f
    f = rho_f*diff(u_vec, t) \
      + rho_f*grad(u_vec)*(u_vec-w_vec)\
      - div(sigma_f(p_c, u_vec, mu_f))

    F_fluid = rho_f/k*inner(u - u0, phi)*dx + rho_f*inner(grad(u)*(u-w), phi)*dx \
       + inner(sigma_f(p, u, mu_f), grad(phi))*dx \
       - inner(f, phi)*dx + inner(div(u), gamma)*dx

    """F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w), phi)*dx
    #F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - ((d-d0)/k)),grad(u)), phi)*dx_f
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx
    F_fluid -= inner(J_(d)*f,phi)*dx"""


    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)
    bcs = [DirichletBC(W.sub(0), u_e, "on_boundary"), \
           DirichletBC(W.sub(1), p_e, "on_boundary")]
    L2_u = []
    L2_p = []
    while t_step <= T:
        u_e.t = t_step
        w_e.t = t_step
        p_e.t = t_step
        d_e.t = t_step
        t.assign(t_step)

        #J = derivative(F, up, phi)
        solve(F_fluid == 0, up, bcs)#, J = J)
        up0.assign(up)
        u_, p_ = up.split(True)
        L2_u.append(errornorm(u_e, u_, norm_type="l2", degree_rise = 2))
        L2_p.append(errornorm(p_e, p_, norm_type="l2", degree_rise = 2))
        t_step += dt


    u_e.t = t_step - dt
    p_e.t = t_step - dt

    E_u.append(np.mean(L2_u))
    E_p.append(np.mean(L2_p))
    h.append(mesh.hmin())


N = [8, 10, 12, 16]
dt = [0.000005]
T = 0.0005
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

N = [32]
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
