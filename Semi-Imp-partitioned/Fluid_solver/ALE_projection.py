from dolfin import *
import sys
import numpy as np

# Local import
from utils.stress import *
from utils.mapping import *
from utils.convergence import convergence
from common.common import solver_parameters
from variationalform.ALE_projection_weakform import ALE_projection_scheme
from solvers.ALE_projection_fluid import ALE_projection_fluid_solver

#Silence FEniCS output
set_log_active(False)

if __name__ == "__main__":

    #Parameters for each numerical case
    common = {"v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 1E-5,          # End time
              "dt": 1E-6,       # Time step
              "N": 8,      #N-points, argument UnitSquareMesh
              "rho": 10,    #
              "mu": 1,
         }

    ######## Convergence Space ########

    #Error storing for Convergence rate
    E_u = []; E_p = []; h = []
    
    N_list = [4, 8, 12]
    #N_list = [2**i for i in range(1, 6)]
    runs = [solver_parameters(common, {"N": i, "dt": 1E-6, "T": 1E-5} ) for i in N_list]

    results = []

    for r in runs:
        vars().update(r)
        print "Solving for N = %d, dt = %g, T = %g" % (r["N"], r["dt"], r["T"])
        results = ALE_projection_scheme(**vars())
        E_u.append(results[0])
        E_p.append(results[1])
        h.append(results[2])

    convergence(E_u, E_p, h, [runs[0]["dt"] ])

    ######## Convergence Time ########
    """
    #Error storing for Convergence rate
    E_u = []; E_p = []; h = []

    dt_list = [0.00005, 0.00004, 0.00002, 0.00001]
    #dt_list = [5E-5/(2**i) for i in range(3)]
    runs = [solver_parameters(common, {"N": 30, "dt": i, "T": 1E-4,\
    "v_deg": 3, "p_deg": 2} ) for i in dt_list]

    results = []

    for r in runs:
        vars().update(r)
        print "Solving for N = %d, dt = %g, T = %g" % (r["N"], r["dt"], r["T"])
        results = ALE_projection_scheme(**vars())
        E_u.append(results[0])
        E_p.append(results[1])
        h.append(results[2])
        #Start simulation

    convergence(E_u, E_p, [runs[0]["N"]], dt_list)
    """
