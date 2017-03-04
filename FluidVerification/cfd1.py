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
    #mesh = refine(mesh)
    #mesh = refine(mesh)
    #mesh = refine(mesh)

    #Parameters for each numerical case
    common = {"mesh": mesh,
              "v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 10,          # End time
              "dt": 0.5,       # Time step
              "rho": 1000.,
              "mu": 1.,
              "Um": 0.2,      #Inflow parameter
              "H": 0.41,      #Hight of tube
         }

    cfd1 = common
    cfd1_2 = solver_parameters(common, {"v_deg": 3, "p_deg": 2})
    cfd1_3 = solver_parameters(common, {"v_deg": 2, "p_deg": 1, "mesh": refine(mesh)})
    cfd1_4 = solver_parameters(common, {"v_deg": 3, "p_deg": 2, "mesh": refine(mesh)})
    case = "cfd1"

    cases  = [cfd1, cfd1_2, cfd1_3, cfd1_4]
    for r in cases:
        vars().update(cfd1)
        Lift, Drag, Time, nel, ndof = mixedformulation(**vars())

        store_results(**vars())
        results(Lift, Drag, Time, nel, ndof, v_deg, p_deg, case)
