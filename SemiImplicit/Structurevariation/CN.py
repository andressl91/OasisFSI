from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world, avg
#from semi_implicit import *

def F_(U):
	return Identity(len(U)) + grad(U)


def J_(U):
	return det(F_(U))


def E(U):
	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))


def S(U,lamda_s,mu_s):
    I = Identity(len(U))
    return 2*mu_s*E(U) + lamda_s*tr(E(U))*I


def Piola1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)


def sigma_f(p, u, d, mu_f):
    return  -p*Identity(len(u)) +\
            mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)


def Structure_setup(d_, w_, v_, p_, phi, gamma, dS2, n, mu_f, body_force, \
            vp_, dx_s, mu_s, rho_s, lamda_s, k, mesh_file, theta, **semimp_namespace):

    delta = 1E10
    #theta = 1.0
    theta = 0.5

    F_solid_linear = rho_s/k*inner(w_["n"] - w_["n-1"], phi)*dx_s \
                   + delta*(1./k)*inner(d_["n"] - d_["n-1"], gamma)*dx_s \
                   - delta*inner(Constant(theta)*w_["n"] \
                   + Constant(1 - theta)*w_["n-1"], gamma)*dx_s

    F_solid_linear -= rho_s*inner(body_force, phi)*dx_s

    # Non-linear part
    F_solid_nonlinear = Constant(theta)*inner(Piola1(d_["n"], lamda_s, mu_s), grad(phi))*dx_s + \
                        Constant(1 - theta)*inner(Piola1(d_["n-1"], lamda_s, mu_s), grad(phi))*dx_s

    u = vp_["tilde"].sub(0)
    p = vp_["n"].sub(1)
    F_solid_nonlinear += inner(Piola1(d_["n"]("-"), lamda_s, mu_s)*n("+") -\
                        J_(d_["tilde"]("+"))*sigma_f(p("+"), u("+"), d_["tilde"]("+"), mu_f) \
                        *inv(F_(d_["tilde"]("+"))).T*n("+"), phi("+"))*dS2(5)

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
