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


def Fluid_tentative_variation(v_tent, v_, d_, beta, dx_f, mu_f,
                              rho_f, k, dS2, **semimp_namespace):

    F_tentative = rho_f/k*J_(d_["tilde"])*inner(v_tent - v_["n-1"], beta)*dx_f

    F_tentative += rho_f*inner(J_(d_["tilde"])*grad(v_tent)*inv(F_(d_["tilde"])) \
                    * (v_["tilde-1"] - (d_["tilde"] - d_["n-1"]) / k), beta)*dx_f
    F_tentative += 2*mu_f*J_(d_["tilde"])*inner(eps(d_["tilde"], v_tent), \
                                                eps(d_["tilde"], beta))*dx_f

    F_tentative -= inner(Constant((0, 0)), beta)*dx_f

    return dict(F_tentative=F_tentative)
