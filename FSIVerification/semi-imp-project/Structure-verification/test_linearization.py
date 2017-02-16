from fenics import AutoSubDomain, DOLFIN_EPS, FunctionSpace, Mesh, \
                    set_log_active, FacetFunction, Constant, \
                    VectorFunctionSpace, MPI, mpi_comm_world
import sys
import numpy as np
from fenicstools import Probes

# Local import 
from stress_tensor import *
from common import *
from weak_form import *

# Set output from FEniCS
set_log_active(False)

# Set ut problem
mesh = Mesh(path.join(rel_path, "mesh", "von_karman_street_FSI_structure.xml"))

# Function space
V = VectorFunctionSpace(mesh, "CG", 2)
VV = V*V

# Get the point [0.2,0.6] at the end of bar
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.2 - DOLFIN_EPS <= coord[1] <= 0.2 + DOLFIN_EPS):
        #print coord
        break

BarLeftSide = AutoSubDomain(lambda x: "on_boundary" and \
                            (((x[0] - 0.2) * (x[0] - 0.2) +
                                (x[1] - 0.2) * (x[1] - 0.2) < 0.0505*0.0505 )
                                and x[1] >= 0.19 \
                                and x[1] <= 0.21 \
                                and x[0] > 0.2)
                            )

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)

# Parameters:
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lambda_ = nu_s*2.*mu_s/(1. - 2.*nu_s)
f = Constant((0, -2.*rho_s))
beta = Constant(0.25)

probe = Probes(coord, V)
def action(wd_, t):
    time.append(t)
    probe(wd_["n"].sub(1))


# Set up different numerical schemes
# TODO: Add options to chose solver and change solver parameters
common = {"space": "mixedspace",
          "E": None,         # Full implicte, not energy conservative
          "T": 0.1,          # End time
          "dt": 0.001,       # Time step
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
runs = [imp] #, ref] #,
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
    # Check if a full simulation has been performed for this
    # simulation set-up, if so load previous simulation results
    if path.exists(r["case_path"]):
        tmp_case_path = r["case_path"]
        f = open(path.join(tmp_case_path, "param.dat", 'w'))
        tmp_param = cPicle.load(f)

        # Only simulations that have been simulated for T=10 is considered
        # as "finished"
        if tmp_param["T"] == 10:
            tmp_dis_x = np.load(path.join(tmp_case_path, "dis_x.np"))
            tmp_dis_y = np.load(path.join(tmp_case_path, "dis_y.np"))
            tmp_time = np.load(path.join(tmp_case_path, "time.np"))
            results.append((tmp_dis_x, tmp_dis_y, tmp_time))
            continue

    # Start simulation
    vars().update(r)
    if r["space"] == "mixedspace":
        problem_mix(**vars())
    elif r["space"] == "singlespace":
        problem_single()
    else:
        print ("Problem type %s is not implemented, only mixedspace "
                + "and singlespace are valid options") % r["space"]
        sys.exit(0)

    displacement = probe.array()
    results.append((displacement[:,0], displacement[:,1], np.array(time)))
    probe.clear()

    # Store results
    if MPI.rank(mpi_comm_world()) == 0:
        makedirs(r["case_path"])
        results[-1][0].dump(path.join(r["case_path"], "dis_x.np"))
        results[-1][1].dump(path.join(r["case_path"], "dis_y.np"))
        results[-1][2].dump(path.join(r["case_path"], "time.np"))
        f = open(path.join(r["case_path"], "param.dat"), "w")
        cPickle.dump(r, f)
        f.close()

#viz(results, runs)
