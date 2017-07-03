from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def sigma_f(p, u, mu_f):
    return  -p*Identity(len(u)) + mu_f*(grad(u) + grad(u).T)

def fluid_setup(v_, p_, n, psi, gamma, dx, mu_f, rho_f, k, dt, theta, **semimp_namespace):

    F_fluid_linear = inner(sigma_f(p_["n"], v_["n"], mu_f), grad(psi))*dx \
        + inner(div(v_["n"]), gamma)*dx

    F_fluid_nonlinear = rho_f*inner(dot(grad(v_["n"]), v_["n"]), psi)*dx

    return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
