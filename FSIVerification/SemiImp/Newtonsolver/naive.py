from dolfin import *

def solver_setup(F_fluid, F_solid, DVP, up_sol, dvp_, **monolithic):

    F_res = F_fluid + F_solid
    a = lhs(F_res)
    L = rhs(F_res)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()

    up_sol.set_operator(A)
    #up_sol.parameters['reuse_factorization'] = True

    return dict(A=A, L=L, up_sol=up_sol)


def linearsolver(A, L, bcs, \
                dvp_, up_sol, T, t, **monolithic):


    b = assemble(L)

    [bc.apply(A, b) for bc in bcs]
    solve(A, dvp_["n"].vector(), b)
    #up_sol.solve(dvp_["n"].vector(), b)

    return dict(dvp_=dvp_, t=t)
