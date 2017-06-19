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
from Fluidvariation.projection import *
from Structurevariation.thetaCN import *
from Domainvariation.domain_update import *
from Newtonsolver.naive import *
from Extrapolation.alfa import *
#exec("from Extrapolation.%s import *" % args.extravari)

#set_log_active(False)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)

# Domains
D = VectorFunctionSpace(mesh_file, "CG", d_deg)
V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)
DVP = MixedFunctionSpace([D, V, P])

# Create functions
dvp_ = {}; d_ = {}; v_ = {}; p_ = {}

for time in ["n", "n-1", "n-2", "tilde"]:
    dvp = Function(DVP)
    dvp_[time] = dvp
    d, v, p = split(dvp)

    d_[time] = d
    v_[time] = v
    p_[time] = p


w = TrialFunction(DVP.sub(0).collapse()) #Mesh velocity
beta = TestFunction(DVP.sub(0).collapse())
w_f = Function(DVP.sub(0).collapse()) #Mesh velocity solution file

v_tilde_n1 = Function(DVP.sub(1).collapse())
v_tilde = Function(DVP.sub(1).collapse())

d_s, v_f, p_f = TrialFunctions(DVP)
phi, psi, gamma = TestFunctions(DVP)

t = 0

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

fluid_sol = LUSolver(solver_method)
solid_sol = LUSolver(solver_method)

vars().update(extrapolate_setup(**vars()))
vars().update(Fluid_tentative_variation(**vars()))
vars().update(Fluid_correction_variation(**vars()))
vars().update(Structure_setup(**vars()))
vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

atol = 1e-6; rtol = 1e-6; max_it = 100; lmbda = 1.0

dvp_res = Function(DVP)
chi = TrialFunction(DVP)

assigner = FunctionAssigner(DVP.sub(2), P)

counter = 0
tic()
while t <= T + 1e-8:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print "Solving for timestep %g" % t

    pre_solve(**vars())
    vars().update(domain_update(**vars()))
    vars().update(Fluid_extrapolation(**vars()))
    vars().update(Fluid_tentative(**vars()))

    #solid_residual_last = 1
    #solid_rel_res_last  = solid_residual_last

    test = 0
    print "BEFORE IT", norm(dvp_["n"].sub(2), "l2")
    while 5  > test:
        vars().update(Fluid_correction(**vars()))
        vars().update(Solid_momentum(**vars()))
        print "IT = %d" % test, norm(dvp_["n"].sub(2), "l2")

        test += 1
        print "BIG ITERATION NUMBER %d" % test

    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
    	dvp_[t_tmp].vector().zero()
    	dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
post_process(**vars())
