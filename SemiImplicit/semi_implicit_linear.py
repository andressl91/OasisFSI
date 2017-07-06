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


print "Semi-implicit methods"
from Fluidvariation.projection_linear import *
from Structurevariation.CN import *
from Domainvariation.domain_update_linear import *
from Newtonsolver.naive_linear import *
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

# Structure Mixedspace
DW = MixedFunctionSpace([D, V])

# Fluid Mixedspace
VP = MixedFunctionSpace([V, P])

# Create functions
dw_ = {}; d_ = {}; w_ = {};
vp_ = {}; v_ = {}; p_ = {};

for time in ["n", "n-1", "n-2", "tilde", "tilde-1"]:
    dw = Function(DW)
    dw_[time] = dw
    d, w = split(dw)

    d_[time] = d
    w_[time] = w

    vp = Function(VP)
    vp_[time] = vp
    v, p = split(vp)

    v_[time] = v
    p_[time] = p

#TrialFunction and TestFunctions Fluid
v, p = TrialFunctions(VP)
psi, eta = TestFunctions(VP)

v_tent = TrialFunction(V)
beta = TestFunction(V)
v_sol = Function(V)

#Trial and TestFunctions Structure
#used for extrapolation of solid deformation
d, w = TrialFunctions(DW)
phi, gamma = TestFunctions(DW)

#Newton iteration function
dw_res = Function(DW)

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


print "SOLVE FOR LINEAR PROBLEM"
atol = 1e-6; rtol = 1e-6; max_it = 100; lmbda = 1.0

counter = 0
tic()
while t <= T + 1e-8:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print "Solving for timestep %g" % t

    pre_solve(**vars())
    vars().update(domain_update(**vars()))
    # Including Fluid_extrapolation gives nan press and veloci
    vars().update(Fluid_extrapolation(**vars()))
    vars().update(Fluid_tentative(**vars()))

    solid_residual_last = 1
    solid_rel_res_last  = solid_residual_last

    test = 0
    while 1 > test:
        vars().update(Fluid_correction(**vars()))
        vars().update(Solid_momentum(**vars()))

        test += 1
        print "BIG ITERATION NUMBER %d" % test

    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        vp_[t_tmp].vector().zero()
        vp_[t_tmp].vector().axpy(1, vp_[times[i+1]].vector())
        dw_[t_tmp].vector().zero()
        dw_[t_tmp].vector().axpy(1, dw_[times[i+1]].vector())

    vp_["tilde-1"].vector().zero()
    vp_["tilde-1"].vector().axpy(1, vp_["tilde"].vector())
    print norm(dw_["n"].sub(0, deepcopy=True))

    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
#_, pp = vp_["n-1"].split(True)
#plot(pp, interactive=True)
#post_process(**vars())
