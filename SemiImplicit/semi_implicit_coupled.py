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
from Fluidvariation.projection_coupled import *
from Structurevariation.coupled import *
from Domainvariation.domain_update_coupled import *
from Newtonsolver.naive_coupled import *
from Extrapolation.alfa_coupled import *

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)

# Domains
V = D = VectorFunctionSpace(mesh_file, "CG", d_deg)
if d_deg != v_deg:
    V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)

# Coupled mixedspace
VPD = MixedFunctionSpace([V, P, D])

# Create functions
vpd_ = {}
d_ = {}
v_ = {}
p_ = {}

for time in ["n", "n-1", "n-2", "n-3", "tilde", "tilde-1"]:
    vpd = Function(VPD)
    vpd_[time] = vpd
    v, p, d = split(vpd)

    v_[time] = v
    p_[time] = p
    d_[time] = d

# Tentative velocity 
v_tent = TrialFunction(V)
beta = TestFunction(V)
v_sol = Function(V)

# Extrapolation of deformation
d = TrialFunction(D)
phi = TestFunction(D)
d_f = Function(D)

# Coupled v, p, d
psi_v, psi_p, psi_d = TestFunctions(VPD)
vpd_res = Function(VPD)

t = 0

# Check for solvers in
for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

fluid_sol = LUSolver(solver_method)
coupled_sol = LUSolver(solver_method)

vars().update(Extrapolate_setup(**vars()))
vars().update(Fluid_tentative_variation(**vars()))
vars().update(Coupled_setup(**vars()))
vars().update(Solver_setup(**vars()))

vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

print "SOLVE FOR LINEAR PROBLEM"
atol = 1e-6; rtol = 1e-6; max_it = 25; lmbda = 0.7

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
    vars().update(Coupled(**vars()))

    times = ["n-3", "n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        vpd_[t_tmp].vector().zero()
        vpd_[t_tmp].vector().axpy(1, vpd_[times[i+1]].vector())

    vpd_["tilde-1"].vector().zero()
    vpd_["tilde-1"].vector().axpy(1, vpd_["tilde"].vector())

    vars().update(after_solve(**vars()))
    counter +=1

simtime = toc()
print "Total Simulation time %g" % simtime
