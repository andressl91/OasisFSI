from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def sigma_f(p, u, mu_f):
    return  -p*Identity(len(u)) + mu_f*(grad(u) + grad(u).T)

def fluid_setup(v_, p_, n, psi, gamma, dx, mu_f, rho_f, k, dt, theta, **semimp_namespace):

	F_fluid_linear = rho_f/k*inner(v_["n"] - v_["n-1"], psi)*dx \
        + Constant(theta)*inner(sigma_f(p_["n"], v_["n"], mu_f), grad(psi))*dx \
		+ Constant(1 - theta)*inner(sigma_f(p_["n-1"], v_["n-1"], mu_f), grad(psi))*dx \
        + Constant(theta)*inner(div(v_["n"]), gamma)*dx \
		+ Constant(1 -theta)*inner(div(v_["n-1"]), gamma)*dx

	F_fluid_nonlinear = Constant(theta)*rho_f*inner(dot(grad(v_["n"]), v_["n"]), psi)*dx \
	+ Constant(1 - theta)*rho_f*inner(dot(grad(v_["n-1"]), v_["n-1"]), psi)*dx

	return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
