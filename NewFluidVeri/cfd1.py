from dolfin import *
import sys
import numpy as np

# Local import
from utils.convergence import convergence
from common.common import solver_parameters
from variationalform.weakform import mixedformulation
from solvers.newtonsolver import Newton_manual

#Silence FEniCS output
#set_log_active(False)

if __name__ == "__main__":

    mesh = Mesh("./mesh/fluid_new.xml")
    #mesh = refine(mesh)

    #Parameters for each numerical case
    common = {"mesh": mesh,
              "v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 7,          # End time
              "dt": 0.2,       # Time step
              "N": 8,      #N-points, argument UnitSquareMesh
              "rho": 1000.,    #
              "mu": 1.,
              "Um": 0.2,
              "H": 0.41,
         }

    cfd1 = common
    runs = [cfd1]

    for r in runs:
        vars().update(r)
        mixedformulation(**vars())

    #convergence(E_u, E_p, h, [runs[0]["dt"] ])

    ######## Convergence Time ########
    """
    #Error storing for Convergence rate
    E_u = []; E_p = []; h = []

    dt_list = [5E-5/(2**i) for i in range(4)]
    runs = [solver_parameters(common, {"N": 40, "dt": i, "T": 2E-4,\
    "v_deg": 3, "p_deg": 2} ) for i in dt_list]

    results = []

    for r in runs:
        vars().update(r)
        print "Solving for N = %d, dt = %g, T = %g" % (r["N"], r["dt"], r["T"])
        results = semi_projection_scheme(**vars())
        E_u.append(results[0])
        E_p.append(results[1])
        h.append(results[2])
        #Start simulation

    convergence(E_u, E_p, [runs[0]["N"] ], dt_list )
    """
