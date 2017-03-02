from dolfin import *
import sys
import numpy as np

# Local import
from utils.results import results
from common.common import solver_parameters
from variationalform.weakform import mixedformulation
from solvers.newtonsolver import Newton_manual

#Silence FEniCS output
#set_log_active(False)

if __name__ == "__main__":

    mesh = Mesh("./mesh/turek1.xml")
    #mesh = refine(mesh)

    #Parameters for each numerical case
    common = {"mesh": mesh,
              "v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 0.04,          # End time
              "dt": 0.02,       # Time step
              "rho": 1000.,
              "mu": 1.,
              "Um": 1.0,      #Inflow parameter
              "H": 0.41,      #Hight of tube
         }

    cfd1 = common

    vars().update(cfd1)
    Lift, Drag, Time, nel, ndof = mixedformulation(**vars())

    case = "cfd2"

    f = open("./results/" + case + "/parameters.txt", 'w')
    for i in range(len(common)):
        f.write('%s = %s \n' % (common.keys()[i], common[common.keys()[i]]))
    f.close()

    results(Lift, Drag, Time, nel, ndof, v_deg, p_deg, case)
