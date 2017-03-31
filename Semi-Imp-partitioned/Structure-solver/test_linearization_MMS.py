from fenics import AutoSubDomain, DOLFIN_EPS, FunctionSpace, Mesh, \
                    set_log_active, FacetFunction, Constant, \
                    VectorFunctionSpace, MPI, mpi_comm_world
import sys
import numpy as np
import time as timer
from fenicstools import Probes

# Local import 
from stress_tensor import *
from common import *
from weak_form import *

# Set output from FEniCS
set_log_active(False)

# Set ut problem
N = [4, 8, 16, 32]
dt = 1e-5

# Nonlinear "reference" simulations
ref = solver_parameters(common, {"E": reference})
imp = solver_parameters(common, {"coupling": "imp"})

# Linear, but not linearized
exp = solver_parameters(common, {"E": explicit, "coupling": "exp"})
center = solver_parameters(common, {"E": explicit, "coupling": "center"})

# Linearization
naive_lin = solver_parameters(common, {"E": naive_linearization})
naive_ab = solver_parameters(common, {"E": naive_ab})
ab_before_cn = solver_parameters(common, {"E": ab_before_cn})
ab_before_cn_higher_order = solver_parameters(common, {"E": ab_before_cn_higher_order})
cn_before_ab = solver_parameters(common, {"E": cn_before_ab})
cn_before_ab_higher_order = solver_parameters(common, {"E": cn_before_ab_higher_order})

for n in N:
    for dt in dt_list:
        mesh = UnitSquareMesh(n, n)
        x = Spatialcoordinates(mesh)

        # Function space
        V = VectorFunctionSpace(mesh, "CG", 2)
        VV = V*V

        u_vec = 
        # Parameters:
        # TODO: make f from d and w

        f = Constant((0, -2.*rho_s))
        beta = Constant(0.25)

        # TODO: check convergence
        def action(wd_, t):
            time.append(t)
            probe(wd_["n"].sub(1))


        # Set up different numerical schemes
        # TODO: Add options to chose solver and change solver parameters
        common = {"space": "mixedspace",
                "E": None,         # Full implicte, not energy conservative
                "T": 10,          # End time
                "dt": 0.01,       # Time step
                "coupling": "CN", # Coupling between d and w
                "init": False      # Solve "exact" three first timesteps
                }

        # Solution set-ups to simulate
        runs = [center] #, imp]
            #ref] #,
            #imp] #,
            #exp] #,
            #naive_lin,
            #naive_ab,
            #ab_before_cn,
            #ab_before_cn_higher_order,
            #cn_before_ab,
            #cn_before_ab_higher_order]

