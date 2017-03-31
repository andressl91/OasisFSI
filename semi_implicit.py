from dolfin import *
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.figure(1)


#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
#ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
# Local import
from Problems.Projection.fsi1 import *
from Utils.convergence import convergence
solver = "ipcs"
exec("from Fluid_solver.%s import *" % solver)

"""
CHECK SOURETERM FOR IPCS, i + 1./2
REMEMBER TURN OFF beta in IPCS FOR IPCS
"""
#Silence FEniCS output
set_log_active(False)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", v_deg)
Q = FunctionSpace(mesh, "CG", p_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh)
nu = Constant(mu_f/rho_f)

# Create functions
u_ = {}; p_ = {};

for time in ["n", "n-1", "n-2"]:
    if time == "n":
        u_[time] = TrialFunction(V)
        u_["sol"] = Function(V)

        p_[time] = TrialFunction(Q)
        p_["sol"] = Function(Q)

    else:
        u_[time] = Function(V)
        p_[time] = Function(Q)

v = TestFunction(V)
q = TestFunction(Q)

vars().update(init(**vars()))
vars().update(create_bcs(**vars()))
vars().update(setup(**vars()))
#vars().update(sourceterm(**vars()))

u_["n-1"].assign(interpolate(u_e, V))
p_["n-1"].assign(interpolate(p_e, Q))

#Doesn't WORK ??
#u_["n-1"] = interpolate(u_e, V)
#p_["n-1"] = interpolate(p_e, Q)

t = dt
#Naive imp for plotting
t_store = []
Err_u = []; Err_p = []
while t <= T + 1e-8:
    t_store.append(t)
    print "Solving for timestep %g" % t
    vars().update(pre_solve(**vars()))
    tentative_velocity_solve(**vars())
    pressure_correction_solve(**vars())
    velocity_update_solve(**vars())

    u_["n-1"].assign(u_["sol"])
    p_["n-1"].assign(p_["sol"])
    #print errornorm(p_e, p_["sol"], norm_type = "l2", degree_rise = 3)
    #p_["n-1"].assign(p_["sol"])
    """
    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        if t_tmp == "n-1":
            u_[t_tmp].vector().zero()
            u_[t_tmp].vector().axpy(1, u_["sol"].vector())

            p_[t_tmp].vector().zero()
            p_[t_tmp].vector().axpy(1, p_["sol"].vector())

        else:
            u_[t_tmp].vector().zero()
            u_[t_tmp].vector().axpy(1, u_[times[i+1]].vector())

            p_[t_tmp].vector().zero()
            p_[t_tmp].vector().axpy(1, p_[times[i+1]].vector())
    """
    #Err_u.append(errornorm(u_e, u_sol, norm_type="l2", degree_rise = 2))
    #Err_p.append(errornorm(p_e, p_sol, norm_type="l2", degree_rise = 2))
    t += dt

h.append(mesh.hmin())
dt_list.append(dt)
t = t - dt
post_process(**vars())
