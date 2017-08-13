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
    return  -p*Identity(len(u)) + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)


def Coupled_setup(d_, v_, p_, psi_v, psi_p, psi_d, dS2, n, mu_f, body_force,
                  dx_s, dx_f, mu_s, rho_s, lamda_s, k, mesh_file, rho_f,
                  **semimp_namespace):

    F_coupled_linear = rho_s / k**2 * inner(d_["n"] - 2*d_["n-1"] + d_["n-2"], psi_d)*dx_s
    F_coupled_linear -= rho_s * inner(body_force, psi_d)*dx_s

    F_coupled_linear = rho_f/k*J_(d_["tilde"])*inner(v_["n"] - v_["tilde"], psi_v)*dx_f
    F_coupled_linear += J_(d_["tilde"])*inner(inv(F_(d_["tilde"])).T*grad(p_["n"]), psi_v)*dx_f
    F_coupled_linear += inner(div(J_(d_["tilde"])*inv(F_(d_["tilde"]))*v_["n"]), psi_p)*dx_f

    # Non-linear part
    F_coupled_nonlinear = Constant(0.5) * inner(Piola1(d_["n"], lamda_s, mu_s), grad(psi_d))*dx_s \
                        + Constant(0.5) * inner(Piola1(d_["n-2"], lamda_s, mu_s), grad(psi_d))*dx_s

    # Impose BC weakly
    F_coupled_nonlinear -= inner(J_(d_["tilde"]("+"))*sigma_f(p_["n"]("+"), v_["tilde"]("+"),
                                                            d_["tilde"]("+"), mu_f) \
                         * inv(F_(d_["tilde"]("+"))).T*n("+"), psi_d("+"))*dS2(5)
    # TODO: This might be better strongly...
    F_coupled_linear += J_(d_["tilde"]("+"))*inner(dot(v_["n"]("+") -
                      -  (d_["n"]("-") - d_["n-1"]("-"))/k, n("+")), psi_p("+"))*dS2(5)


    return dict(F_coupled_linear=F_coupled_linear, F_coupled_nonlinear=F_coupled_nonlinear)
