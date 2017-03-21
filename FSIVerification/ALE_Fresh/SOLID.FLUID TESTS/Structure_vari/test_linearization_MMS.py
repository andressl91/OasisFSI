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

for n in N:
    mesh = UnitSquareMesh(n, n)
    # Function space
    V = VectorFunctionSpace(mesh, "CG", 2)
    VV = V*V

    # Get the point [0.2,0.6] at the end of bar

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

    # Solution set-ups to simulate
    runs = [ref] #, imp]
        #ref] #,
        #imp] #,
        #exp] #,
        #naive_lin,
        #naive_ab,
        #ab_before_cn,
        #ab_before_cn_higher_order,
        #cn_before_ab,
        #cn_before_ab_higher_order]
    #runs = [exp]
    results = []
    for r in runs:
        time = []
        load = False
    # Check if a full simulation has been performed for this
    # simulation set-up, if so load previous simulation results
    if path.exists(r["case_path"]):
        tmp_case_path = r["case_path"]
        file_ = open(path.join(tmp_case_path, "param.dat"), "r")
        tmp_param = cPickle.load(file_)
        file_.close()
        tmp_time = np.load(path.join(tmp_case_path, "time.np"))

        # Only simulations that have been simulated for T=10 is considered
        # as "finished"
        if tmp_param["T"] == 10 and tmp_time[-1] >= 10 - 1e-12:
            print "Load solution from", r["name"]
            load = True
            tmp_dis_x = np.load(path.join(tmp_case_path, "dis_x.np"))
            tmp_dis_y = np.load(path.join(tmp_case_path, "dis_y.np"))
            tmp_time = np.load(path.join(tmp_case_path, "time.np"))
            results.append((tmp_dis_x, tmp_dis_y, tmp_time))
            continue

    # Start simulation
    vars().update(r)
    t0 = timer.time()
    try:
        if r["space"] == "mixedspace":
            problem_mix(**vars())
        elif r["space"] == "singlespace":
            problem_single()
        else:
            print ("Problem type %s is not implemented, only mixedspace "
                    + "and singlespace are valid options") % r["space"]
            sys.exit(0)
    except Exception as e:
        print "Problems for solver", r["name"]
        print "Unexpected error:", sys.exc_info()[0]
        print e
        print "Move on the the next solver"
    r["number_of_cores"] = MPI.max(mpi_comm_world(), MPI.rank(mpi_comm_world())) + 1
    r["solution_time"] = MPI.sum(mpi_comm_world(), timer.time() - t0) / (r["number_of_cores"])

    displacement = probe.array()
    probe.clear()

    # Store results
    if MPI.rank(mpi_comm_world()) == 0 and not load:
        results.append((displacement[0,:], displacement[1,:], np.array(time)))
        if not path.exists(r["case_path"]):
            makedirs(r["case_path"])
        results[-1][0].dump(path.join(r["case_path"], "dis_x.np"))
        results[-1][1].dump(path.join(r["case_path"], "dis_y.np"))
        results[-1][2].dump(path.join(r["case_path"], "time.np"))
        file_ = open(path.join(r["case_path"], "param.dat"), "w")
        cPickle.dump(r, file_)
        file_.close()

#viz(results, runs)
