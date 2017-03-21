from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym
#from semi_implicit import *

def F_(u):
    I = Identity(2)
    return I

def J_(u):
    return det(F_(u))


def setup(u_, p_, d_, w_, u_t, u_tilde, u0_tilde, phi, v, q, dx, mu_f, rho_f, k, **semimp_namespace):
    #Tentative Velocity
    #k = Constant(dt)
    Fu_tent = (rho_f/k)*inner(u_t - u_["n-1"], phi)*dx
    Fu_tent += rho_f*inner(dot(grad(u_t), u0_tilde), phi)*dx
    Fu_tent += 2.*mu_f*inner(sym(grad(u_t)), sym(grad(phi)))*dx
    #Fu_tent += 2.*mu_f*inner(sym(grad(u_t)), sym(grad(phi)))*dx

    # Pressure update
    Fu_corr = rho_f/k*inner(u_["n"] - u_tilde, v)*dx
    Fu_corr -= inner(p_["n"], div(v))*dx
    Fu_corr -= inner(q, div(u_["n"]))*dx
    #Fu_corr += J_(d_["n"])*inner(grad(u_["n"]), inv(F_(d_["n"])).T)*q*dx
    """
    Fu_tent = (rho_f/k)*J_(d_["n-1"])*inner(u_t - u_["n-1"], phi)*dx
    Fu_tent += rho_f*J_(d_["n-1"])*inner(inv(F_(d_["n-1"]))*dot(grad(u_t), u0_tilde - w_["n-1"]), phi)*dx
    Fu_tent += J_(d_["n-1"])*mu_f*inner(grad(u_t)*inv(F_(d_["n-1"])), grad(phi)*inv(F_(d_["n-1"])) )*dx

    # Pressure update
    Fu_corr = rho_f/k*J_(d_["n"])*inner(u_["n"] - u_tilde, v)*dx
    Fu_corr -= p_["n"]*J_(d_["n"])*inner(inv(F_(d_["n"])).T, grad(v))*dx
    Fu_corr += J_(d_["n"])*inner(grad(u_["n"]), inv(F_(d_["n"])).T)*q*dx
    #Fu_corr += J_(d_["n"])*inner(grad(u_["n"]), inv(F_(d_["n"])).T)*q*dx
    """
    return dict(Fu_tent=Fu_tent, Fu_corr = Fu_corr)

def tentative_velocity_solve(Fu_tent, u_tilde, bcs_u, **semimp_namespace):
    a = lhs(Fu_tent)
    L = rhs(Fu_tent)
    solve(a == L, u_tilde, bcs_u)
    """
    A = assemble(lhs(Fu_tent), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_tent))
    [bc.apply(A, b) for bc in bcs_u]
    solve(A, u_tilde.vector(), b)
    """
def pressure_correction_solve(Fu_corr, up_, bcs_up, **semimp_namespace):
    a = lhs(Fu_corr)
    L = rhs(Fu_corr)
    solve(a == L, up_["n"], bcs_up)
    """
    A = assemble(lhs(Fu_corr), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_corr))
    [bc.apply(A, b) for bc in bcs_up]
    solve(A, up_["n"].vector(), b)
    """
