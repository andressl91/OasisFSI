from fenics import *
import numpy as np
set_log_active(False)
import argparse
from argparse import RawTextHelpFormatter

def parse():
    parser = argparse.ArgumentParser(description="MMS of ALE\n",\
     formatter_class=RawTextHelpFormatter, \
      epilog="############################################################################\n"
      "Example --> python ALE_MMS.py -source_term 0 -var_form 1\n"
      "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-var_form",  type=int, help="Which form to use  --> Default=0, no mapping", default=0)
    group.add_argument("-source_term",  type=int, help="Which source_term to use  --> Default=0, no mapping", default=0)

    return parser.parse_args()
args = parse()
var_form = args.var_form
source_term = args.source_term


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
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
I = Identity(2)

def solver(N, dt, T):
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q
    n = FacetNormal(mesh)

    up = Function(W)
    u, p = split(up)

    phi,gamma = TestFunctions(W)

    up0 = Function(W)
    u0, p0 = split(up0)
    w = Function(V)
    d = Function(V)
    d0 = Function(V)

    k = Constant(dt)
    t_step = dt

    mu_f = 10
    rho_f = 1

    u_e = Expression(("sin(x[1])*cos(t)","sin(x[0])*cos(t)"), t = dt)
    p_e = Expression("2*sin(t)", t = dt)
    w_e = Expression(("sin(x[1])*cos(t)","sin(x[0])*cos(t)"), t = dt)
    d_e = Expression(("sin(x[1])*sin(t)","sin(x[0])*sin(t)"), t = dt)

    w = interpolate(w_e,V)
    d = interpolate(d_e,V)
    u0 = interpolate(u_e,V)
    t = Constant(dt)

    u_x = sin(x[1])*cos(t)
    u_y = sin(x[0])*cos(t)
    p_c = 2*sin(t)

    u_vec = as_vector([u_x, u_y])
    w_vec = as_vector([u_x, u_y])

    # Create right hand side f
    if source_term ==0:
        f = rho_f*diff(u_vec, t) \
          + rho_f*grad(u_vec)*(u_vec-w_vec)\
          - div(sigma_f(p_c, u_vec, mu_f))

    elif source_term == 1:
        f = J_(d)*rho_f*diff(u_vec, t) \
        + J_(d)*rho_f*inv(F_(d))*dot((u_vec-w_vec),grad(u_vec))\
        - div(J_(d)*sigma_f_new(u_vec, p_c, d, mu_f)*inv(F_(d)).T)

    elif source_term == 2:
        f = J_(d)*rho_f*diff(u_vec, t) \
        + J_(d)*rho_f*grad(u_vec)*inv(F_(d))*(u_vec-w_vec)\
        - div(J_(d)*sigma_f_new(u_vec, p_c, d, mu_f)*inv(F_(d)).T)

    if var_form == 0: #no mappings
        F_fluid = rho_f/k*inner(u - u0, phi)*dx + rho_f*inner(grad(u)*(u-w), phi)*dx \
           + inner(sigma_f(p, u, mu_f), grad(phi))*dx \
           - inner(f, phi)*dx - inner(div(u), gamma)*dx

    elif var_form == 1: # The richter way,
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - w),grad(u)), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx
        F_fluid -= inner(J_(d)*f, phi)*dx

    elif var_form == 2: # Written with tensor and grad on the left side.
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx
        F_fluid -= inner(J_(d)*f, phi)*dx

    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)
    bcs = [DirichletBC(W.sub(0), u_e, "on_boundary"), \
           DirichletBC(W.sub(1), p_e, "on_boundary")]
    L2_u = []
    L2_p = []


    psi = TrialFunction(W)
    while t_step <= T:
        u_e.t = t_step
        w_e.t = t_step
        p_e.t = t_step
        d_e.t = t_step
        t.assign(t_step)

        J = derivative(F_fluid, up, psi)
        solve(F_fluid == 0, up, bcs,J=J,solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":20,"relaxation_parameter":1.0}})
        up0.assign(up)

        if var_form == 0 :
            w.vector()[:] *= float(k)
            ALE.move(mesh,w)
            mesh.bounding_box_tree().build(mesh)

        u_, p_ = up.split(True)

        L2_u.append(errornorm(u_e, u_, norm_type="l2", degree_rise = 2))
        L2_p.append(errornorm(p_e, p_, norm_type="l2", degree_rise = 2))
        t_step += dt


    u_e.t = t_step - dt
    p_e.t = t_step - dt

    E_u.append(np.mean(L2_u))
    E_p.append(np.mean(L2_p))
    h.append(mesh.hmin())



N = [4, 6, 8,10]
dt = [0.000001]
T = 0.0001
E_u = [];  E_p = []; h = []

for n in N:
    for t in dt:
        print "Solving for t = %g, N = %d" % (t, n)
        solver(n, t, T)

print "Checking Convergence in Space P2-P1"

for i in E_u:
    print "Errornorm Velocity L2", i


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

"""
print "Checking Convergence in time"

N = [16]
dt = [0.005, 0.004, 0.002, 0.001]
T = 0.04
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

    print "Convergence Pressure", r_u"""
