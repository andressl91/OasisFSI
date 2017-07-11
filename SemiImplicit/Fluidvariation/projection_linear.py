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
    beta, dx_f, mu_f, rho_f, k, dt, dS2, **semimp_namespace):

    #Reuse of TrialFunction w, TestFunction psi
    #used in extrapolation assuming same degree

    F_tentative = rho_f/k*J_(d_["tilde"])*inner(v_tent - v_["n-1"], beta)*dx_f

    F_tentative += rho_f*inner(J_(d_["tilde"])*grad(v_tent)*inv(F_(d_["tilde"])) \
                    * (v_["tilde-1"] - 1./k*(d_["tilde"] - d_["n-1"])), beta)*dx_f

    F_tentative += J_(d_["tilde"])*inner(2*mu_f*eps(d_["tilde"], v_tent), eps(d_["tilde"], beta))*dx_f

    F_tentative -= inner(Constant((0, 0)), beta)*dx_f

    return dict(F_tentative=F_tentative)

def Fluid_correction_variation(v, p, v_, d_, vp_, dw_, psi, eta, dx_f, \
    mu_f, rho_f, k, dt, n, dS2, **semimp_namespace):

    # Pressure update
    F_correction = rho_f/k*J_(d_["tilde"])*inner(v - v_["tilde"], psi)*dx_f

    # This gives no pressure gradient thorugh the pipe
    #F_correction -= J_(d_["tilde"])*inner(p*inv(F_(d_["tilde"])).T, grad(psi))*dx_f
    F_correction += J_(d_["tilde"])*inner(inv(F_(d_["tilde"])).T*grad(p), psi)*dx_f

    F_correction += inner(div(J_(d_["tilde"])*inv(F_(d_["tilde"]))*v), eta)*dx_f

    #F_correction += J_(d_["tilde"]("+"))*inner(dot(v("+") - \
    #                (d_["n"]("-") - d_["n-1"]("-"))/k, n("+")), eta("+"))*dS2(5)

    return dict(F_correction=F_correction)
