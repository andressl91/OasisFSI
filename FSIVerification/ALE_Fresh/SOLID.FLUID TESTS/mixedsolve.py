from dolfin import *
import sys
import numpy as np

import matplotlib.pyplot as plt

# Local import
#from Problems.Coupled.mms import *
from Problems.Coupled.TG2D import *
from Utils.convergence import convergence
from Fluid_solver.Coupled.coupled import *

# Silence FEniCS output
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
        V = VectorFunctionSpace(mesh, "CG", v_deg, constrained_domain = constrained_domain)
        Q = FunctionSpace(mesh, "CG", p_deg, constrained_domain = constrained_domain)
        VQ = MixedFunctionSpace([V, Q])

        # Define coefficient
        nu = Constant(mu_f/rho_f)
        k = Constant(dt)
        n = FacetNormal(mesh)
        vars().update(initiate(**vars()))

        # Create functions
        u_ = {}; p_ = {}; up_ = {}
        for time in ["n", "n-1", "n-2"]:
            up = Function(VQ)
            up_[time] = up
            u, p = split(up) #.split(deepcopy=True)
            u_[time] = u
            p_[time] = p

        v, q = TestFunctions(VQ)

        vars().update(create_bcs(**vars()))
        #vars().update(setup(**vars()))

        F =  rho_f/k*inner(u_["n"] - u_["n-1"], v)*dx
        F += rho_f*inner(dot(u_["n"], grad(u_["n"])), v)*dx
        F += mu*inner(grad(u_["n"]), grad(v))*dx
        F -= inner(p_["n"], div(v))*dx
        F -= inner(q, div(u_["n"]))*dx

        #vars().update(sourceterm(**vars()))
        u_init = interpolate(u_e, V)
        p_init = interpolate(p_e, Q)
        assign(up_["n-1"].sub(0), u_init)
        assign(up_["n-1"].sub(1), p_init)

        t = 0

        tic()
        while t < T:
            print "Solving for t = %g" % t
            t += dt
            #vars().update(pre_solve(**vars()))
            solve(F==0, up_["n"])
            u_num, p_num = up_["n-1"].split(True)
            assign(up_["n-1"].sub(0), u_num)
            assign(up_["n-1"].sub(1), p_num)
            #up_["n-1"].assign(up_["n"])

        print "END TIME %g" % toc()
        #u_num, p_num = up1.split(True)
        u_num, p_num = up_["n-1"].split(True)
        u_e.t = t
        p_e.t = t
        E_u.append(errornorm(u_e, u_num, degree_rise=3))
        E_p.append(errornorm(p_e, p_num, degree_rise=3))
        h.append(mesh.hmin())
        #post_process(**vars())

check = Dt if len(Dt) > 1 else h

for i in range(len(E_u) - 1):
    r_u = np.log(E_u[i+1]/E_u[i])/np.log(check[i+1]/check[i])
    r_p = np.log(E_p[i+1]/E_p[i])/np.log(check[i+1]/check[i])
    print r_u, r_p
#plt.show()
"""
if len(N) > len(Dt):
    dt_list = [dt_list[0]]
else:
    h = [h[0]]
convergence(E_u, E_p, h, dt_list)
"""
