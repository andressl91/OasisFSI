from dolfin import *
#from semi_implicit import *

def F_(d):
    I = Identity(len(d)) + grad(d)
    return I

def J_(d):
    return det(F_(d))


def fluid_setup(u_, p_, d_, w_, v, q, dx_f, mu_f, rho_f, k, **semimp_namespace):
    #Tentative Velocity

    Fu_tent = (1./k)*J_(d_["n"])*inner(u_["n"] - u_["n-1"], v)*dx_f
    Fu_tent += J_(d_["n"])*inner(inv(F_(d_["n"]))*dot(grad(u_["n"]), u_["tilde_n-1"] - w_["n"]), v)*dx_f
    Fu_tent += J_(d_["n"])*mu_f/rho_f*inner(grad(u_["n"])*inv(F_(d_["n"])), grad(v)*inv(F_(d_["n"])) )*dx_f

    # Pressure update
    Fp_corr = inner(F_(d_["n"])*grad(p_["n"]), grad(q))*dx_f + (1./k)*div(J_(d_["n"])*F_(d_["n"])*u_["sol"])*q*dx_f
    # Velocity update
    Fu_upt = J_(d_["n"])/k*inner(u_["n"] - u_["sol"], v)*dx_f + inner(grad(p_["sol"]), v)*dx_f \


    return dict(Fu_tent = Fu_tent, Fp_corr = Fp_corr, Fu_upt = Fu_upt)

def tentative_velocity_solve(Fu_tent, u_, bcs_u_tent, **semimp_namespace):
    #a = lhs(Fu_tent)
    #L = rhs(Fu_tent)
    #solve(a == L, u_["tilde_n"], bcs_u_tent)

    A = assemble(lhs(Fu_tent), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_tent))
    [bc.apply(A, b) for bc in bcs_u_tent]
    solve(A, u_["sol"].vector(), b)

def pressure_correction_solve(Fp_corr, p_, bcs_p, **semimp_namespace):
    #a = lhs(Fu_corr)
    #L = rhs(Fu_corr)
    #solve(a == L, p_["sol"], bcs_p)

    A = assemble(lhs(Fp_corr), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fp_corr))
    [bc.apply(A, b) for bc in bcs_p]
    solve(A, p_["sol"].vector(), b)


def velocity_update_solve(Fu_upt, u_, bcs_u_corr, **semimp_namespace):
    #a = lhs(Fu_corr)
    #L = rhs(Fu_corr)
    #solve(a == L, u_["sol"], bcs_u_corr)

    A = assemble(lhs(Fu_upt), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(Fu_upt))
    [bc.apply(A, b) for bc in bcs_u_corr]
    solve(A, u_["sol"].vector(), b)
