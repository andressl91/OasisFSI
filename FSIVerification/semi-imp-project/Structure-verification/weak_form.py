from fenics import Constant, DirichletBC, Function, TrialFunction, split, \
                    inner, dx, grad, TestFunctions
import sys

from stress_tensor import *
from solvers import *

def problem_mix(T, dt, E, coupling, VV, boundaries, rho_s, lambda_, mu_s, f,
                bcs, **Solid_namespace):
    # Temporal parameters
    t = 0
    k = Constant(dt)

    # Split problem to two 1.order differential equations
    psi, phi = TestFunctions(VV)

    # Functions, wd is for holding the solution
    d_ = {}; w_ = {}; wd_ = {}
    for time in ["n", "n-1", "n-2", "n-3"]:
        if time == "n" and E not in [None, reference]:
            tmp_wd = Function(VV)
            wd_[time] = tmp_wd
            wd = TrialFunction(VV)
            w, d = split(wd)
        else:
            wd = Function(VV)
            wd_[time] = wd
            w, d = split(wd)

        d_[time] = d
        w_[time] = w

    # Time derivative
    if coupling == "center":
        G = rho_s/(2*k) * inner(w_["n"] - w_["n-2"], psi)*dx
    else:
        G = rho_s/k * inner(w_["n"] - w_["n-1"], psi)*dx

    # Stress tensor
    G += inner(Piola2(d_, w_, k, lambda_, mu_s, E_func=E), grad(psi))*dx

    # External forces, like gravity
    G -= rho_s*inner(f, psi)*dx

    # d-w coupling
    if coupling == "CN":
        G += inner(d_["n"] - d_["n-1"] - k*0.5*(w_["n"] + w_["n-1"]), phi)*dx
    elif coupling == "imp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n"], phi)*dx
    elif coupling == "exp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n-1"], phi)*dx
    elif coupling == "center":
        G += innter(d_["n"] - d_["n-2"] - 2*k*w["n-1"], phi)*dx
    else:
        print "The coupling %s is not implemented, 'CN', 'imp', and 'exp' are the only valid choices."
        sys.exit(0)

    # Solve
    if E in [None, reference]:
        solver_nonlinear(G, d_, w_, wd_, bcs, T, dt, **Solid_namespace)
    else:
        solver_linear(G, d_, w_, wd_, bcs, T, dt, **Solid_namespace)

# TODO: Implement a version with only d
