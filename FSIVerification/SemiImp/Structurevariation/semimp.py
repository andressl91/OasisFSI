from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world
#from semi_implicit import *

def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def E(U, U_tilde):
	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))

def S(U, U_tilde, lamda_s,mu_s):
    I = Identity(len(U))
    return 2*mu_s*E(U) + lamda_s*tr(E(U))*I


def Piola1(U, U_tilde, lamda_s,mu_s):
    I = Identity(len(U))
    E = 0.5*(F_(U).T*F_(U_tilde) - I)
    S = 2.*mu_s*E + lamda_s*tr(E)*I

    return F_(U_tilde)*S


def structure_setup(d, v, p, d_, v_, p_, phi, psi, gamma, dS, mu_f, n,\
            dx_s, dx_f, mu_s, rho_s, lamda_s, k, mesh_file, theta, **semimp_namespace):

    delta = 1
    #First order
    """
    dvdt = 1./k*(v - v_["n-1"])
    dudt = 1./(2*k)*(d - d_["n-2"])
    J_tilde = J_(d_["n-1"])
    F_tilde = F_(d_["n-1"])
    d_tilde = d_["n-1"]
    v_tilde = v_["n-1"]
    """

    #  Second Order
    dvdt = 1./(2*k)*(3*v - 4*v_["n-1"] + v_["n-2"])
    dudt = 1./(2*k)*(3*d - 4*d_["n-1"] + d_["n-2"])
    J_tilde = 2.*J_(d_["n-1"]) - J_(d_["n-2"])
    F_tilde = 2.*F_(d_["n-1"]) - F_(d_["n-2"])
    d_tilde = 2.*d_["n-1"] - d_["n-2"]
    

    F_solid = rho_s*inner(dvdt, psi)*dx_s \
                   + delta*inner(dudt, phi)*dx_s \
    				   - delta*inner(v, phi)*dx_s

    F_solid += inner(Piola1(d, d_tilde, lamda_s, mu_s), grad(psi))*dx_s

    return dict(F_solid = F_solid)
