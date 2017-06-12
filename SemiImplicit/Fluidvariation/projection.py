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

def D_U(d, u):
	return 1./2*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)
    #return grad(v)*inv(F_(d))

def Fluid_tentative_variation(v_, p_, d_, dvp_, w, w_f, v_tilde_n1, \
psi, beta, gamma, dx_f, mu_f, rho_f, k, dt, **semimp_namespace):

	d_tent = dvp_["tilde"].sub(0, deepcopy=True)

	v_n1   = dvp_["n-1"].sub(1, deepcopy=True)
	d_n = dvp_["tilde"].sub(0, deepcopy=True)
	d_n1 = dvp_["n-1"].sub(0, deepcopy=True)

	#Reuse of TrialFunction w, TestFunction beta
	#used in extrapolation assuming same degree

	F_tentative = rho_f/k*J_(d_tent)*inner(w - v_n1, beta)*dx_f
	F_tentative += rho_f*inner(J_(d_tent)*grad(w)*inv(F_(d_tent)) \
             	* (v_tilde_n1 - 1./k*(d_n - d_n1)), beta)*dx_f

	F_tentative += J_(d_tent)*inner(2*mu_f*D_U(d_tent, w), D_U(d_tent, beta))*dx_f

	#F_tentative = rho_f*inner(J_(d_tent)*grad(w)*inv(F_(d_tent)) \
	#             * (v_tilde_n1 - w_f), beta)*dx_f

	#F_tentative += J_(d_tent)*inner(sigma_f_u(w, d_tent, mu_f), grad(beta))*dx_f


	F_tentative -= inner(Constant((0, 0)), beta)*dx_f

	return dict(F_tentative=F_tentative)

def Fluid_correction_variation(v_f, p_f, v_, p_, d_, dvp_, psi, gamma, dx_f, \
    mu_f, rho_f, k, dt, n, dS, **semimp_namespace):

	#Updating domain
	"""
	# Pressure update
	F_correction = rho_f/k*J_(d_["n"])*inner(v_f - v_["tilde"], psi)*dx_f
	F_correction -= p_f*J_(d_["n"])*inner(inv(F_(d_["n"])).T, grad(psi))*dx_f
	F_correction += J_(d_["n"])*inner(grad(v_f), inv(F_(d_["n"])).T)*gamma*dx_f

	#Use newly computed "n" from step 3.2m first time "tilde"
	F_correction += J_(d_["n"]("+"))*dot(v_f("+") - \
	(d_["n"]("-")-d_["n-1"]("-"))/k, n("+"))*gamma("+")*dS(5)

	"""
	#No update domain
	# Pressure update
	F_correction = rho_f/k*J_(d_["tilde"])*inner(v_f - v_["tilde"], psi)*dx_f
	F_correction -= p_f*J_(d_["tilde"])*inner(inv(F_(d_["tilde"])).T, grad(psi))*dx_f
	F_correction += J_(d_["tilde"])*inner(grad(v_f), inv(F_(d_["tilde"])).T)*gamma*dx_f

	#Use newly computed "n" from step 3.2m first time "tilde"
	F_correction += J_(d_["tilde"]("+"))*dot(v_f("+") - \
	(d_["n"]("-")-d_["n-1"]("-"))/k, n("+"))*gamma("+")*dS(5)

	#F_correction += dot(v_f("+") - \
	#(d_["n"]("-")-d_["n-1"]("-"))/k, n("+"))*gamma("+")*dS(5)
	return dict(F_correction=F_correction)
