from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def sigma_f(u, p, nu):
    return 2.0*nu*sym(grad(u)) - p*Identity(len(u))


def setup(u_, p_, n, u_sol, p_sol, v, q, dx, ds, mu_f, rho_f, k, **semimp_namespace):

    U = 0.5*(u_["n"] + u_["n-1"])

    # Tentative velocity step
    nu = Constant(mu_f/rho_f)
    beta = Constant(1)

    # Pressure update
    Fu_tent = 1./k*inner(u_["n"] - u_["n-1"], v)*dx \
    + inner(dot(u_["n-1"], grad(u_["n-1"])), v)*dx \
    + inner(sigma_f(U, p_["n-1"], nu), sym(grad(v)) )*dx \
    - beta*nu*inner(dot(n, grad(U).T), v)*ds + inner(p_["n-1"]*n ,v)*ds

    # Velocity update
    Fp_corr = dot(grad(p_["n"]), grad(q))*dx - dot(grad(p_["n-1"]), grad(q))*dx  \
    + 1./k*div(u_sol)*q*dx

    Fu_upt = inner(u_["n"] - u_sol, v)*dx \
    + k*dot(grad(p_sol - p_["n-1"]), v)*dx

    return dict(Fu_tent=Fu_tent, Fp_corr = Fp_corr, Fu_upt = Fu_upt)


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
