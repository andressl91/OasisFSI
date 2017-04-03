from dolfin import *
import sys
import numpy as np


def sigma_f_new(u,p,d,mu_f):
	return -p*Identity(len(u)) + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))

def S(U,lamda_s,mu_s):
    I = Identity(len(U))
    return 2*mu_s*E(U) + lamda_s*tr(E(U))*I

def Piola1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)

# Get user input
from Utils.argpar import *
args = parse()

# Mesh refiner
exec("from Problems.%s import *" % args.problem)
if args.refiner != None:
    for i in range(args.refiner):
        mesh = refine(mesh)

update_variables = {}

#Update argparser input, due to how files are made in problemfiles
for key in args.__dict__:
    if args.__dict__[key] != None:
        update_variables[key] = args.__dict__[key]

vars().update(update_variables)


# Import variationalform and solver
print args.solver
exec("from Fluidvariation.%s import *" % args.fluidvari)
exec("from Structurevariation.%s import *" % args.solidvari)
exec("from Newtonsolver.%s import *" % args.solver)
#Silence FEniCS output
set_log_active(False)

#Domains
D = VectorFunctionSpace(mesh_file, "CG", d_deg)
V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)

DVP = MixedFunctionSpace([D, V, P])

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)
#nu = Constant(mu_f/rho_f)

# Create functions

dvp_ = {}; d_ = {}; v_ = {}; p_ = {}

for time in ["n", "n-1", "n-2"]:
    dvp = Function(DVP)
    dvp_[time] = dvp
    d, v, p = split(dvp)

    d_[time] = d
    v_[time] = v
    p_[time] = p

phi, psi, gamma = TestFunctions(DVP)
t = 0

F_fluid_linear = (rho_f/k)*inner(J_(d_["n"])*(v_["n"] - v_["n-1"]), psi)*dx_f
F_fluid_linear -= inner(div(J_(d_["n"])*inv(F_(d_["n"]))*v_["n"]), gamma)*dx_f
F_fluid_linear += inner(J_(d_["n"])*sigma_f_new(v_["n"], p_["n"], d_["n"], mu_f)*inv(F_(d_["n"])).T, grad(psi))*dx_f
F_fluid_nonlinear = rho_f*inner(J_(d_["n"])*grad(v_["n"])*inv(F_(d_["n"]))*(v_["n"] - ((d_["n"]-d_["n-1"])/k)), psi)*dx_f

delta = 1E10
F_solid_linear = rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s
F_solid_nonlinear = inner(Piola1(0.5*(d_["n"] + d_["n-1"]), lamda_s, mu_s), grad(psi))*dx_s

#Deformation relation to velocity
F_solid_linear += delta*((1./k)*inner(d_["n"] - d_["n-1"],phi)*dx_s - inner(0.5*(v_["n"] + v_["n-1"]), phi)*dx_s)

# laplace
d_exp, _u, _p = dvp_["n"].split(True)
F_expro = inner(grad(d_exp), grad(phi))*dx_f - inner(Constant((0, 0)), phi)*dx_f

d_w    = DirichletBC(D, ((0.0, 0.0)), boundaries, 2)
d_i   = DirichletBC(D, ((0.0, 0.0)), boundaries, 3)
d_o  = DirichletBC(D, ((0.0, 0.0)), boundaries, 4)
d_c  = DirichletBC(D, ((0.0, 0.0)), boundaries, 6)
d_bar = DirichletBC(D, ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

d_bcs = [d_w, d_i, d_o, d_c, d_bar]

vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

atol = 1e-8; rtol = 1e-8; max_it = 100; lmbda = 1.0

dvp_res = Function(DVP)
chi = TrialFunction(DVP)

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

#up_sol = LUSolver(mpi_comm_world(),solver_method)
up_sol = LUSolver(solver_method)
#up_sol.parameters["same_nonzero_pattern"] = True
#up_sol.parameters["reuse_factorization"] = True


counter = 0
tic()
while t <= T + 1e-8:
	t += dt

	if MPI.rank(mpi_comm_world()) == 0:
	    print "Solving for timestep %g" % t

	pre_solve(**vars())
	vars().update(newtonsolver(**vars()))

	solve(F_expro == 0, d_exp, d_bcs)
	d = DVP.sub(0).dofmap().collapse(mesh_file)[1].values()
	dvp_["n"].vector()[d] = d_exp.vector()

	times = ["n-2", "n-1", "n"]
	for i, t_tmp in enumerate(times[:-1]):
		dvp_[t_tmp].vector().zero()
		dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())
	vars().update(after_solve(**vars()))

	counter +=1

print "TIME SPENT!!!", toc()

post_process(**vars())
