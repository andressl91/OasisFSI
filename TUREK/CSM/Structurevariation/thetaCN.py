from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world
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

def structure_setup(d_, v_, phi, psi, n, rho_s, \
            dx, mu_s, lamda_s, k, mesh_file, theta, **semimp_namespace):

	delta = 1E10

	F_solid_linear = rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx \
	               + delta*(1/k)*inner(d_["n"] - d_["n-1"], phi)*dx \
					   - delta*inner(Constant(theta)*v_["n"] + Constant(1 - theta)*v_["n-1"], phi)*dx

	gravity = Constant((0, -2*rho_s))
	F_solid_linear -= inner(gravity, psi)*dx

	F_solid_nonlinear = inner(Piola1(Constant(theta)*d_["n"] + Constant(1 - theta)*d_["n-1"], lamda_s, mu_s), grad(psi))*dx

	return dict(F_solid_linear = F_solid_linear, F_solid_nonlinear = F_solid_nonlinear)
