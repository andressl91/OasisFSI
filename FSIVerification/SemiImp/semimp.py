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

exec("from Fluidvariation.%s import *" % args.fluidvari)
exec("from Structurevariation.%s import *" % args.solidvari)
exec("from Extrapolation.%s import *" % args.extravari)
exec("from Newtonsolver.%s import *" % args.solver)
#Silence FEniCS output
#set_log_active(False)

#Domains
D = VectorFunctionSpace(mesh_file, "CG", d_deg)
V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)


# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)


if args.extravari == "biharmonic" or args.extravari == "biharmonic2":
    print "Biharmonic"
    DVP = MixedFunctionSpace([D, V, P, D])

    dvp_ = {}; d_ = {}; v_ = {}; p_ = {}; w_ = {}

    for time in ["n", "n-1", "n-2", "n-3"]:
        dvp = Function(DVP)
        dvp_[time] = dvp
        d, v, p, w = split(dvp)

        d_[time] = d
        v_[time] = v
        p_[time] = p
        w_[time] = w

    phi, psi, gamma, beta = TestFunctions(DVP)

else :
    DVP = MixedFunctionSpace([D, V, P])

    # Create functions

    dvp_ = {}; d_ = {}; v_ = {}; p_ = {}

    for time in ["n", "n-1", "n-2", "n-3"]:
        dvp = Function(DVP)
        dvp_[time] = dvp
        d_f, v_f, p_f = split(dvp)

        d_[time] = d_f
        v_[time] = v_f
        p_[time] = p_f

    d, v, p = TrialFunctions(DVP)
    phi, psi, gamma = TestFunctions(DVP)

t = 0

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

up_sol = LUSolver(solver_method)
#up_sol.parameters["same_nonzero_pattern"] = True
up_sol.parameters["reuse_factorization"] = True


vars().update(fluid_setup(**vars()))
vars().update(structure_setup(**vars()))
#vars().update(extrapolate_setup(**vars()))
vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

counter = 0
tic()
while t <= T + 1e-8:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print "Solving for timestep %g" % t

    pre_solve(**vars())
    vars().update(linearsolver(**vars()))

    times = ["n-3", "n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
    	dvp_[t_tmp].vector().zero()
    	dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())
    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
post_process(**vars())
