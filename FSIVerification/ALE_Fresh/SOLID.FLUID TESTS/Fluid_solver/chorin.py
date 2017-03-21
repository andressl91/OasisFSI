from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad, dx


def setup(u_, p_, v, q, dx, mu_f, rho_f, nu, k, **semimp_namespace):

    # Tentative velocity step
    Fu_tent = (1./k)*inner(u_["n"] - u_["n-1"], v)*dx \
    + inner(dot(u_["n-1"], grad(u_["n-1"])), v)*dx \
    + nu*inner(grad(u_["n"]), grad(v))*dx

    # Pressure update
    Fp_corr = 1/rho_f*inner(grad(p_["n"]), grad(q))*dx + (1./k)*div(u_["sol"])*q*dx

    # Velocity update
    Fu_upt = 1./k*inner(u_["n"] - u_["sol"], v)*dx + 1./rho_f*inner(grad(p_["sol"]), v)*dx

    return dict(Fu_tent = Fu_tent, Fp_corr = Fp_corr, Fu_upt = Fu_upt)

def tentative_velocity_solve(Fu_tent, u_, bcs_u, **semimp_namespace):
    a = lhs(Fu_tent)
    L = rhs(Fu_tent)
    solve(a == L, u_["sol"], bcs_u)
    """
    A = assemble(lhs(Fu_tent), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_tent))
    [bc.apply(A, b) for bc in bcs_u]
    solve(A, u_tilde.vector(), b)
    """
def pressure_correction_solve(Fp_corr, p_, bcs_p, **semimp_namespace):
    a = lhs(Fp_corr)
    L = rhs(Fp_corr)
    solve(a == L, p_["sol"], bcs_p)
    """
    A = assemble(lhs(Fu_corr), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_corr))
    [bc.apply(A, b) for bc in bcs_up]
    solve(A, up_["n"].vector(), b)
    """
def velocity_update_solve(Fu_upt, bcs_u, u_, **semimp_namespace):
    a = lhs(Fu_upt)
    L = rhs(Fu_upt)
    solve(a == L, u_["sol"], bcs_u)
