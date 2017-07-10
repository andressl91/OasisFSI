from dolfin import *
import sys
import numpy as np

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
exec("from Structurevariation.%s import *" % args.solidvari)
exec("from Newtonsolver.%s import *" % args.solver)
#Silence FEniCS output
#set_log_active(False)

#Domains
V = VectorFunctionSpace(mesh_file, "CG", v_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)


VV = MixedFunctionSpace([V, V])

# Create functions

dv_ = {}; d_ = {}; v_ = {}

for time in ["n", "n-1", "n-2", "n-3"]:
    dv = Function(VV)
    dv_[time] = dv
    d, v= split(dv)

    d_[time] = d
    v_[time] = v

phi, psi = TestFunctions(VV)

t = 0

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i


dv_sol = LUSolver(solver_method)

vars().update(structure_setup(**vars()))
vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

dummy_file = XDMFFile(mpi_comm_world(), path + "/dummy.xdmf")
for tmp_t in [dummy_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 0
    tmp_t.parameters["rewrite_function_mesh"] = False


atol = 1e-6; rtol = 1e-6; max_it = 100; lmbda = 1.0

dv_res = Function(VV)


counter = 0
tic()
while t <= T + 1e-8:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print "Solving for timestep %g" % t

    pre_solve(**vars())
    vars().update(newtonsolver(**vars()))

    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
    	dv_[t_tmp].vector().zero()
    	dv_[t_tmp].vector().axpy(1, dv_[times[i+1]].vector())
    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
post_process(**vars())
