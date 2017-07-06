from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad


def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def sigma_f_u(u,d,mu_f):
    return  mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def sigma_f_p(p, u):
    return -p*Identity(len(u))

def D_U(d, v):
	#return 1/2.*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)
	return 1./2*grad(v)*inv(F_(d))

def fluid_setup(d, v , p, v_, p_, d_, n, psi, gamma, dx_f, ds, mu_f, rho_f, k, dt, v_deg, theta, **semimp_namespace):

	#First Order
	"""
	dvdt = 1./k*(d - v_["n-1"])
	dudt = 1./(2*k)*(d_["n-1"] - d_["n-3"])
	J_tilde = J_(d_["n-1"])
	F_tilde = F_(d_["n-1"])
	d_tilde = d_["n-1"]
	"""

	#Second Order
	dvdt = 1./(2*k)*(3.*v - 4.*v_["n-1"] + v_["n-2"])
	dudt = 1./(2*k)*(3.*d_["n-1"] - 4.*d_["n-2"] + d_["n-3"])
	J_tilde = 2.*J_(d_["n-1"]) - J_(d_["n-2"])
	F_tilde = 2.*F_(d_["n-1"]) - F_(d_["n-2"])
	d_tilde = 2.*d_["n-1"] - d_["n-2"]


	F_fluid = rho_f*inner(J_tilde*dvdt, psi)*dx_f
	F_fluid += rho_f*inner(J_tilde*grad(v)*inv(F_tilde)*(v_["n-1"] - dudt), psi)*dx_f
	F_fluid += J_tilde*inner(2*mu_f*D_U(d_tilde, v), D_U(d_tilde, psi))*dx_f
	F_fluid -= inner(J_tilde*p*inv(F_tilde).T, grad(psi))*dx_f

	#F_fluid += inner(div(J_tilde*inv(F_tilde)*v), gamma)*dx_f #Check 2.13 paper
	F_fluid += inner(J_tilde*grad(v), inv(F_tilde).T*gamma)*dx_f #Check 2.13 paper


	#Not a must
	#djdt = 1./(2*k)*(3.*J_(d_["n-1"]) - 4.*J_(d_["n-2"]) + J_(d_["n-3"]) )
	#F_fluid += rho_f/2.*inner(djdt*v, psi)*dx_f
	#F_fluid += rho_f/2.*inner(div(J_(d_["n-1"])*inv(F_tilde)\
	#		*(v_["n-1"] - dudt))*v, psi)*dx_f


	return dict(F_fluid = F_fluid)
