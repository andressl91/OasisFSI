from dolfin import *
import sys
import numpy as np

from Problems.cfd1 import *
from Fluidvariation.fluid_coupled import *
from Structurevariation.CN_mixed import *
from Solver.newtonsolver import *

#Silence FEniCS output
set_log_active(False)

E_u = []; E_p = [];

#TODO: Fix mesh import, dolfin import overwrites mesh imported from
#problemfile with mesh.module

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

vars().update(create_bcs(**vars()))
vars().update(fluid_setup(**vars()))
vars().update(structure_setup(**vars()))
#vars().update(init(**vars()))

F_lin = F_fluid_linear + F_solid_linear
F_nonlin = F_fluid_nonlinear + F_solid_nonlinear

chi = TrialFunction(DVP)
J_linear    = derivative(F_lin, dvp_["n"], chi)
J_nonlinear = derivative(F_nonlin, dvp_["n"], chi)

A_pre = assemble(J_linear)
A = Matrix(A_pre)
b = None

F = F_lin + F_nonlin
t = 0

atol = 1e-8; rtol = 1e-8; max_it = 100; lmbda = 1.0

dvp_res = Function(DVP)
chi = TrialFunction(DVP)

for i in ["mumps", "superlu_dist", "default"]:
    if has_lu_solver_method(i):
        solver_method = i

up_sol = LUSolver(solver_method)
up_sol.parameters["same_nonzero_pattern"] = True
up_sol.parameters["reuse_factorization"] = True #Maby, maby not doesnt do mutch
tic()
while t <= T + 1e-8:
    t += dt
    print "Solving for timestep %g" % t
    pre_solve(**vars())
    newtonsolver(**vars())

    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
    	dvp_[t_tmp].vector().zero()
    	dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    vars().update(after_solve(**vars()))

print "TIME SPENT!!!", toc()
t = t - dt
post_process(**vars())
