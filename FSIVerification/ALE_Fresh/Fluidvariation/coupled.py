from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def sigma_f(u, p, nu):
    return 2.0*nu*sym(grad(u)) - p*Identity(len(u))


def setup(u_, p_, d_, n, psi, gamma, dx_s, ds, mu_f, rho_f, nu, k, **semimp_namespace):

    F_fluid_linear = (rho_f/k)*inner(J_(d_["n"])*(u_["n"] - u_["n-1"]), phi)*dx_f
    F_fluid_linear -= inner(div(J_(d_["n"])*inv(F_(d_["n"]))*u_["n"]), gamma)*dx_f
    F_fluid_linear += inner(J_(d_["n"])*sigma_f_new(u_["n"], p_["n"], d_, mu_f)*inv(F_(d_["n"])).T, grad(phi))*dx_f
    F_fluid_nonlinear = rho_f*inner(J_(d_["n"])*grad(u_["n"])*inv(F_(d_["n"]))*(u_["n"] - ((d_-d_)/k)), phi)*dx_f
    if v_deg == 1:
        F_fluid -= beta*h*h*inner(J_(d_["n"])*inv(F_(d_["n"]).T)*grad(p), grad(gamma))*dx_f

        print "v_deg",v_deg

    return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)


def tentative_velocity_solve(Fu_tent, u_sol, bcs_u, **semimp_namespace):
    a = lhs(Fu_tent)
    L = rhs(Fu_tent)
    solve(a == L, u_sol, bcs_u)
    """
    A = assemble(lhs(Fu_tent), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_tent))
    [bc.apply(A, b) for bc in bcs_u]
    solve(A, u_tilde.vector(), b)
    """
def pressure_correction_solve(Fp_corr, p_sol, bcs_p, **semimp_namespace):
    a = lhs(Fp_corr)
    L = rhs(Fp_corr)
    solve(a == L, p_sol, bcs_p)
    """
    A = assemble(lhs(Fu_corr), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_corr))
    [bc.apply(A, b) for bc in bcs_up]
    solve(A, up_["n"].vector(), b)
    """
def velocity_update_solve(Fu_upt, bcs_u, u_sol, **semimp_namespace):
    a = lhs(Fu_upt)
    L = rhs(Fu_upt)
    solve(a == L, u_sol, bcs_u)
