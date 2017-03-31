from dolfin import *
import sys
import numpy as np

# Local import
from utils.results import results
from utils.store_results import store_results
from common.common import solver_parameters
from variationalform.weakform import mixedformulation
from solvers.newtonsolver import Newton_manual

#Silence FEniCS output
#set_log_active(False)

if __name__ == "__main__":

    mesh = Mesh("./mesh/turek1.xml")
    mesh = refine(mesh)
    mesh = refine(mesh)

    #Parameters for each numerical case
    common = {"mesh": mesh,
              "v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 10,          # End time
              "dt": 0.01,       # Time step
              "rho": 1000.,
              "mu": 1.,
              "Um": 1.0,      #Inflow parameter
              "H": 0.41,      #Hight of tube
         }

    cfd2 = common
    cfd2_2 = solver_parameters(common, {"v_deg": 2, "p_deg": 1, "dt": 0.001})
    cfd2_3 = solver_parameters(common, {"v_deg": 3, "p_deg": 2})
    cfd2_4 = solver_parameters(common, {"v_deg": 3, "p_deg": 2, "dt": 0.001})
    cfd2_5 = solver_parameters(common, {"v_deg": 2, "p_deg": 1, "mesh": refine(mesh)})
    cfd2_6 = solver_parameters(common, {"v_deg": 3, "p_deg": 2, "mesh": refine(mesh)})
    case = "cfd2"

    cases  = [cfd2, cfd2_2, cfd2_3, cfd2_4, cfd2_5, cfd2_6]
    for r in cases:
        vars().update(r)
        Lift, Drag, Time, nel, ndof = mixedformulation(**vars())

        store_results(**vars())
        results(Lift, Drag, Time, nel, ndof, v_deg, p_deg, case)
