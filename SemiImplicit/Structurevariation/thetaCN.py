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

def sigma_f(p, u, d, mu_f):
    return  -p*Identity(len(u)) +\
	        mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def Structure_setup(d_, v_, p_, phi, psi, gamma, dS, n, mu_f, \
            dx_s, dx_f, mu_s, rho_s, lamda_s, k, mesh_file, theta, **semimp_namespace):

	delta = 1E10
	theta = 0.5


	F_solid_linear = rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s \
	               + delta*(1./k)*inner(d_["n"] - d_["n-1"], phi)*dx_s \
   				   - delta*inner(Constant(theta)*v_["n"] + Constant(1 - theta)*v_["n-1"], phi)*dx_s

	#F_solid_nonlinear = inner(Piola1(Constant(theta)*d_["n"] + Constant(1 - theta)*d_["n-1"], lamda_s, mu_s), grad(psi))*dx_s
	F_solid_nonlinear = Constant(theta)*inner(Piola1(d_["n"], lamda_s, mu_s), grad(psi))*dx_s \
	 			      + Constant(1 - theta)*inner(Piola1(d_["n-1"], lamda_s, mu_s), grad(psi))*dx_s

	#Test
	#F_solid_nonlinear += inner((Piola1(d_["n"]("+"), lamda_s, mu_s) - \
	#J_(d_["n"]("-"))*sigma_f(p_["n"]("-"),v_["tilde"]("-"), d_["n"]("-"), mu_f) \
	#*inv(F_(d_["n"]("-"))).T)*n("+"), phi("+"))*dS(5)

	F_solid_nonlinear -= inner(J_(d_["n"]("+"))*sigma_f(p_["n"]("+"),v_["n"]("+"), d_["n"]("+"), mu_f) \
	*inv(F_(d_["n"]("+"))).T*n("-"), phi("-"))*dS(5)

	#Org
	#F_solid_nonlinear += inner((Piola1(d_["n"]("-"), lamda_s, mu_s) - \
	#J_(d_["n"]("+"))*sigma_f(p_["n"]("+"),v_["tilde"]("+"), d_["n"]("+"), mu_f) \
	#*inv(F_(d_["n"]("+"))).T)*n("-"), phi)*dS(5)
	return dict(F_solid_linear = F_solid_linear, F_solid_nonlinear = F_solid_nonlinear)
