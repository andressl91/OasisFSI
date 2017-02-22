from fenics import *
import mshr
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

I = Identity(2)
def F_(U):
	return (Identity(2) + grad(U))

def J_(U):
	return det(F_(U))
def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U,lamda_s,mu_s):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)

def solver(N, dt, T):
    mesh = UnitSquareMesh(N, N)

    x = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = V*V
    n = FacetNormal(mesh)

    ud = Function(W)
    u, d = split(ud)

    phi, psi = TestFunctions(W)
    ud0 = Function(W)
    u0, d0 = split(ud0)

    k = Constant(dt)
    t_step = dt

    mu_s = 1
    rho_s = 1
    lamda_s = 1

    #u_e = Expression(("x[0]*x[0]*x[0]+4*t*t*t","x[1]*x[1]*x[1]+4*t*t*t"), degree = 2,  t = 0)
    #d_e = Expression(("x[0]*x[0]*x[0]+t*t*t*t","x[1]*x[1]*x[1]+t*t*t*t"), degree = 2, t = 0)
    d_e = Expression(("cos(x[1])*sin(t)",
                  "cos(x[0])*sin(t)"
                 ), degree = 2, t = 0)

    u_e = Expression(("cos(x[1])*cos(t)",
                  "cos(x[0])*cos(t)"
                 ), degree = 2, t = 0)
    u0 = interpolate(u_e, V)
    d0 = interpolate(d_e,V)
    t = Constant(dt)

    """u_x = x[0]*x[0]*x[0]+4*t*t*t
    u_y = x[1]*x[1]*x[1]+4*t*t*t
    d_x = x[0]*x[0]*x[0]+t*t*t*t
    d_y = x[1]*x[1]*x[1]+t*t*t*t
    """

    d_x = cos(x[1])*sin(t)
    d_y = cos(x[0])*sin(t)

    u_x = cos(x[1])*cos(t)
    u_y = cos(x[0])*cos(t)

    u_vec = as_vector([u_x, u_y])
    d_vec = as_vector([d_x, d_y])
    # Create right hand side f
    f1 =rho_s*diff(u_vec, t) - div(P1(d_vec,lamda_s,mu_s))
    #f2 = diff(d_vec, t) - u_vec # is zero when d and u is created to be zero

    delta = 1
    F_structure = (rho_s/k)*inner(u-u0,phi)*dx
    F_structure -= inner(div(P1(d,lamda_s,mu_s)), phi)*dx
    F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx - inner(u,psi)*dx)
    F_structure -= inner(f1, phi)*dx #+ inner(f2, psi)*dx

    bcs = [DirichletBC(W.sub(0), u_e, "on_boundary"), \
           DirichletBC(W.sub(1), d_e, "on_boundary")]

    L2_u = []
    L2_d = []

    u_file = XDMFFile(mpi_comm_world(), "Structure_MMS_results/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), "Structure_MMS_results/d.xdmf")

    for tmp_t in [u_file, d_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 1
        tmp_t.parameters["rewrite_function_mesh"] = False

    d_diff = Function(V)
    u_diff = Function(V)

    while t_step <= T:
        u_e.t = t_step
        d_e.t = t_step
        t.assign(t_step)

        solve(F_structure == 0, ud, bcs,solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        u_, d_ = ud.split(True)
        u0.assign(u_); d0.assign(d_)

        d_e_diff = project(d_e, V)
        d_diff.vector().zero()
        d_diff.vector().axpy(1, d_e_diff.vector())
        d_diff.vector().axpy(-1, d_.vector())
        d_diff.rename("p_diff", "Error in p for each time step")
        d_file << d_diff

        u_e_diff = project(u_e, V)
        u_diff.vector().zero()
        u_diff.vector().axpy(1, u_e_diff.vector())
        u_diff.vector().axpy(-1, u_.vector())
        u_diff.rename("u_diff", "Error in u for each time step")
        u_file << u_diff

        L2_u.append(errornorm(u_e, u_, norm_type="l2", degree_rise = 3))
        L2_d.append(errornorm(d_e, d_, norm_type="l2", degree_rise = 3))

        t_step += dt

    E_u.append(np.mean(L2_u))
    E_d.append(np.mean(L2_d))
    h.append(mesh.hmin())

"""
N = [4,8,16,32,64]
dt = [1.0E-6]
T = 1.0E-5
E_u = [];  E_d = []; h = []

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

for i in E_d:
    print "Errornorm Deformation L2", i

print

for i in range(len(E_d) - 1):
    r_d = np.log(E_d[i+1]/E_d[i]) / np.log(h[i+1]/h[i])

    print "Convergence Deformation", r_d

"""
print "Checking Convergence in time"

N = [64]
dt = [8.0E-4, 4E-4, 2E-4, 1E-4, 0.5E-4]
T = 1.0E-2
E_u = [];  E_d = []; h = []
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

for i in E_d:
    print "Errornorm Deformation L2", i

print

for i in range(len(E_d) - 1):
    r_d = np.log(E_d[i+1]/E_d[i]) / np.log(dt[i+1]/dt[i])

    print "Convergence Deformation", r_d
