from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad


def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def eps(d, u):
    return  1./2*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def sigma_f_p(p, u):
    return -p*Identity(len(u))

def D_U(d, v):
    return 1./2*grad(v)*inv(F_(d))

def Fluid_tentative_variation(v_tent, v_, p_, d_, dw_, vp_, v, \
    beta, dx_f, mu_f, rho_f, k, dt, v_n1, v_tent_n1, d_n1, d_tent, **semimp_namespace):

	#Reuse of TrialFunction w, TestFunction psi
	#used in extrapolation assuming same degree

    # FIXME: v_n1 is always zero as it is never updated, deepcopy!!
	#v_n1 = vp_["n-1"].sub(0) #, deepcopy=True)
	#v_tilde_n1 = vp_["tilde-1"].sub(0) #, deepcopy=True)
	#d_n1 = dw_["n-1"].sub(0) #, deepcopy=True)
	#d_tilde = dw_["tilde"].sub(0) #, deepcopy=True)

	F_tentative = rho_f/k*J_(d_tent)*inner(v_tent - v_n1, beta)*dx_f

	F_tentative += rho_f*inner(J_(d_tent)*grad(v_tent)*inv(F_(d_tent)) \
                            * (v_tent_n1 - 1./k*(d_tent - d_n1)), beta)*dx_f

	F_tentative += J_(d_tent)*inner(2*mu_f*eps(d_tent, v_tent),
                                  eps(d_tent, beta))*dx_f

	F_tentative -= inner(Constant((0, 0)), beta)*dx_f

	return dict(F_tentative=F_tentative)

def Fluid_correction_variation(v, p, v_, d_, vp_, dw_, psi, eta, dx_f, \
    mu_f, rho_f, k, dt, n, dS, **semimp_namespace):

	# Pressure update
	F_correction = rho_f/k*J_(d_["tilde"])*inner(v - v_["tilde"], psi)*dx_f

	# This gives no pressure gradient thorugh the pipe
	F_correction -= J_(d_["tilde"])*inner(p*inv(F_(d_["tilde"])).T, grad(psi))*dx_f

	# This makes the simulation diverge
    #F_correction += J_(d_["tilde"])*inner(inv(F_(d_["tilde"])).T*grad(p), psi)*dx_f

	F_correction += inner(div(J_(d_["tilde"])*inv(F_(d_["tilde"]))*v), eta)*dx_f

	F_correction += J_(d_["tilde"]("+"))*dot(v("+") - \
	(d_["n"]("-")-d_["n-1"]("-"))/k, n("+"))*eta("+")*dS(5)


	#F_correction += J_(d_["tilde"]("+"))*dot(v("+") - \
	#(dw_["n"].sub(0, deepcopy=True)("-")-dw_["n-1"].sub(0, deepcopy=True)("-"))/k, n("-"))*eta("+")*dS(5)

	#Use newly computed "n" from step 3.2m first time "tilde"


	return dict(F_correction=F_correction)
