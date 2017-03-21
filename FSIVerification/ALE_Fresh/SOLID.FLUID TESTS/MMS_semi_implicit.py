from dolfin import *
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.figure(1)


#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
#ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
# Local import
from Problems.Projection.TG2D import *
from Utils.convergence import convergence
solver = "chorin"
exec("from Fluid_solver.%s import *" % solver)

def sigma_f(p_, u_, mu_f):
    return -p_*Identity(2) + 2*mu_f*sym(grad(u_))

"""
CHECK SOURETERM FOR IPCS, i + 1./2
REMEMBER TURN OFF beta in IPCS FOR IPCS
"""
#Silence FEniCS output
set_log_active(False)

E_u = []; E_p = []; h = []; dt_list = []

for n_ in N:
    for dt in Dt:

        mesh = RectangleMesh(Point(0, 0), Point(2, 2), n_, n_)

        class PeriodicDomain(SubDomain):
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
                return bool((near(x[0], 0) or near(x[1], 0)) and
                      (not ((near(x[0], 0) and near(x[1], 2)) or
                            (near(x[0], 2) and near(x[1], 0)))) and on_boundary)

            def map(self, x, y):
                if near(x[0], 2) and near(x[1], 2):
                    y[0] = x[0] - 2.0
                    y[1] = x[1] - 2.0
                elif near(x[0], 2):
                    y[0] = x[0] - 2.0
                    y[1] = x[1]
                else:
                    y[0] = x[0]
                    y[1] = x[1] - 2.0

        constrained_domain = PeriodicDomain()

        # Define function spaces (P2-P1)
        V = VectorFunctionSpace(mesh, "CG", v_deg, constrained_domain = constrained_domain)
        Q = FunctionSpace(mesh, "CG", p_deg, constrained_domain = constrained_domain)

        # Define coefficients
        k = Constant(dt)
        n = FacetNormal(mesh)
        nu = Constant(mu_f/rho_f)

        # Create functions
        u_ = {}; p_ = {}; up_ = {};
        for time in ["n", "n-1", "n-2"]:
            if time == "n":
                u_[time] = TrialFunction(V)
                p_[time] = TrialFunction(Q)
                u_["sol"] = Function(V)
                p_["sol"] = Function(Q)
            else:
                u_[time] = Function(V)
                p_[time] = Function(Q)

        v = TestFunction(V)
        q = TestFunction(Q)

        vars().update(initiate(**vars()))
        vars().update(create_bcs(**vars()))
        vars().update(setup(**vars()))
        #vars().update(sourceterm(**vars()))


        u_["n-1"].assign(project(u_e, V))
        p_["n-1"].assign(project(p_e, Q))

        #Doesn't WORK ??
        #u_["n-1"] = project(u_e, V)
        #p_["n-1"] = project(p_e, Q)

        t = 0

        while t < T:
            t += dt
            print "Solving for timestep %g" % t

            vars().update(pre_solve(**vars()))
            tentative_velocity_solve(**vars())
            pressure_correction_solve(**vars())
            velocity_update_solve(**vars())
            u_["n-1"].assign(u_["sol"])
            p_["n-1"].assign(p_["sol"])

        h.append(mesh.hmin())
        dt_list.append(dt)
        u_e.t = t
        p_e.t = t
        E_u.append(errornorm(u_e, u_["sol"], degree_rise=3))
        E_p.append(errornorm(p_e, p_["sol"], degree_rise=3))
        h.append(mesh.hmin())

        #post_process(**vars())
"""
check = Dt if len(Dt) > 1 else h

for i in range(len(E_u) - 1):
    r_u = np.log(E_u[i+1]/E_u[i])/np.log(check[i+1]/check[i])
    r_p = np.log(E_p[i+1]/E_p[i])/np.log(check[i+1]/check[i])
    print r_u, r_p
"""
if len(N) > len(Dt):
    dt_list = [dt_list[0]]
else:
    h = [h[0]]
convergence(E_u, E_p, h, dt_list)
