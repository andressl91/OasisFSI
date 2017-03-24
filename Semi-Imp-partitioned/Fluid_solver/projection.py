from dolfin import *
import sys
import numpy as np

# Local import
from utils.convergence import convergence
from common.common import solver_parameters
from variationalform.projection_weakform import semi_projection_scheme
from solvers.projection_fluid import projection_fluid_solver

#Silence FEniCS output
set_log_active(False)

if __name__ == "__main__":

    #Parameters for each numerical case
    common = {"v_deg": 2,    #Velocity degree
              "p_deg": 1,    #Pressure degree
              "T": 2E-5,          # End time
              "dt": 1E-6,       # Time step
              "rho": 1,    #
              "mu": 1,
         }

    vars().update(common)
    #u_x = "x[1]"
    #u_y = "x[0]"
    #p_c = "2"

    u_x = "cos(x[1]) + sin(t_)"
    u_y = "cos(x[0]) + cos(t_)"
    p_c = "cos(x[0]) + cos(t_)"

    #u_x = "cos(x[1])*sin(t_)"
    #u_y = "cos(x[0])*sin(t_)"
    #p_c = "cos(x[0])*sin(t_)"

    ######## Convergence Space ########
    E_u = []; E_p = []; h = []

    N_list = [4, 8, 12, 16]
    runs = [solver_parameters(common, {"N": i} ) for i in N_list]

    results = []

    for r in runs:
        vars().update(r)
        print "Solving for N = %d, dt = %g, T = %g" % (r["N"], r["dt"], r["T"])
        results = semi_projection_scheme(**vars())
        E_u.append(results[0])
        E_p.append(results[1])
        h.append(results[2])
        #Start simulation


    convergence(E_u, E_p, h, [runs[0]["dt"] ])

    ######## Convergence Time ########
    E_u = []; E_p = []; h = []

    #dt_list = [1E-1]
    dt_list = [5E-2, 4E-2, 2E-2, 1E-2]
    runs = [solver_parameters(common, {"N": 40, "dt": i, "T": 2E-1,\
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
