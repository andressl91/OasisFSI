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

def A_E(J, v, d, rho_f, mu_f, psi, dx_f):
    return rho_f*inner(J*grad(v)*inv(F_(d))*v, psi)*dx_f \
        + inner(J*sigma_f_u(v, d, mu_f)*inv(F_(d)).T, grad(psi))*dx_f


def fluid_setup(v_, p_, d_, n, psi, gamma, dx_f, ds, mu_f, rho_f, k, dt, v_deg, theta, **semimp_namespace):

    J_theta = theta*J_(d_["n"]) + (1 - theta)*J_(d_["n-1"])
    d_cn = Constant(theta)*d_["n"] + Constant(1 - theta)*d_["n-1"]
    v_cn = Constant(theta)*v_["n"] + Constant(1 - theta)*v_["n-1"]
    p_cn = Constant(theta)*p_["n"] + Constant(1 - theta)*p_["n-1"]


    F_fluid_linear = rho_f/k*inner(J_theta*(v_["n"] - v_["n-1"]), psi)*dx_f
    F_fluid_nonlinear =  rho_f*inner(J_(d_cn)*grad(v_cn)*inv(F_(d_cn))* \
                         (0.5*(3*v_["n-1"] - v_["n-2"]) -\
                         (d_["n"]-d_["n-1"])/k), psi)*dx_f

    F_fluid_nonlinear += inner(J_(d_["n"])*sigma_f_p(p_["n"], d_["n"])*inv(F_(d_["n"])).T, grad(psi))*dx_f
    F_fluid_nonlinear += inner(J_(d_cn)*sigma_f_u(v_cn, d_cn, mu_f)*inv(F_(d_cn)).T, grad(psi))*dx_f
    #OrgF_fluid_nonlinear +=inner(div(J_(d_["n"])*inv(F_(d_["n"]))*v_["n"]), gamma)*dx_f
    F_fluid_nonlinear +=inner(div(J_(d_cn)*inv(F_(d_cn))*v_cn), gamma)*dx_f


    return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
