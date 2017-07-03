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
exec("from Fluidvariation.%s import *" % args.fluidvari)
exec("from Newtonsolver.%s import *" % args.solver)

#Domains
V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)


# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)

VP = MixedFunctionSpace([V, P])

# Create functions

vp_ = {}; v_ = {}; p_ = {}

for time in ["n", "n-1", "n-2", "n-3"]:
    vp = Function(VP)
    vp_[time] = vp
    v, p = split(vp)
    v_[time] = v
    p_[time] = p

psi, gamma = TestFunctions(VP)

t = 0

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

up_sol = LUSolver(solver_method)

vars().update(fluid_setup(**vars()))
vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

atol = 1e-6; rtol = 1e-6; max_it = 100; lmbda = 1.0

vp_res = Function(VP)
chi = TrialFunction(VP)

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
    	vp_[t_tmp].vector().zero()
    	vp_[t_tmp].vector().axpy(1, vp_[times[i+1]].vector())
    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
post_process(**vars())
