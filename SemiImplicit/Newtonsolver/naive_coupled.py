from dolfin import *

def Solver_setup(F_extrapolate, F_tentative, F_coupled_linear,
                 F_coupled_nonlinear, fluid_sol, coupled_sol,
                 vpd_, VPD, **monolithic):

    a = lhs(F_extrapolate)
    A_extra = assemble(a, keep_diagonal=True)
    A_extra.ident_zeros()

    a = lhs(F_tentative)
    A_tent = assemble(a, keep_diagonal=True)
    A_tent.ident_zeros()

    F_coupled = F_coupled_linear + F_coupled_nonlinear
    chi = TrialFunction(VPD)
    Jac_coupled = derivative(F_coupled, vpd_["n"], chi)

    return dict(A_extra=A_extra, A_tent=A_tent, Jac_coupled=Jac_coupled,
                F_coupled=F_coupled)


def Fluid_extrapolation(F_extrapolate, A_extra, VPD, d_f, vpd_, bcs_w,
                        mesh_file, **semimp_namespace):
    L = rhs(F_extrapolate)
    b = assemble(L)
    [bc.apply(A_extra, b) for bc in bcs_w]

    print "Solving fluid extrapolation"
    solve(A_extra, d_f.vector(), b)

    tr = VPD.sub(2).dofmap().collapse(mesh_file)[1].values()
    vpd_["n"].vector()[tr] = d_f.vector()

    return {}


def Fluid_tentative(F_tentative, A_tent, VPD, vpd_, bcs_tent,
                    v_sol, mesh_file, **semimp_namespace):

    L = rhs(F_tentative)
    b = assemble(L)
    [bc.apply(A_tent, b) for bc in bcs_tent]

    print "Solving tentative velocity"
    solve(A_tent, v_sol.vector(), b)

    tr = VPD.sub(0).dofmap().collapse(mesh_file)[1].values()
    vpd_["tilde"].vector()[tr] = v_sol.vector()

    return {}



def Coupled(F_coupled, Jac_coupled, bcs_coupled, vpd_, VPD, coupled_sol,
            vpd_res, rtol, atol, max_it, T, t, lmbda, **monolithic):

    Iter = 0
    coupled_residual = 1
    coupled_rel_res = coupled_residual
    print "Coupled\n"

    while coupled_rel_res > rtol and coupled_residual > atol and Iter < max_it:
        # Comment inn for speed-up
        """
        if Iter % 6  == 0:# or (last_rel_res < rel_res and last_residual < residual):
            print "assebmling new JAC"
            A = assemble(Jac_coupled, keep_diagonal = True)
            [bc.apply(A) for bc in bcs_coupled]
            A.ident_zeros()
            coupled_sol.set_operator(A)

        #[bc.apply(b, vpd_["n"].vector()) for bc in bcs_coupled]
        #coupled_sol.solve(vpd_res.vector(), b)
        """
        # Comment out for speed-up
        A = assemble(Jac_coupled, keep_diagonal=True)
        A.ident_zeros()
        coupled_sol.set_operator(A)
        ###

        b = assemble(-F_coupled)

        [bc.apply(A, b, vpd_["n"].vector()) for bc in bcs_coupled]
        coupled_sol.solve(vpd_res.vector(), b)

        vpd_["n"].vector().axpy(lmbda, vpd_res.vector())
        [bc.apply(vpd_["n"].vector()) for bc in bcs_coupled]
        coupled_rel_res = norm(vpd_res, 'l2')
        coupled_residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.8e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
                    % (Iter, coupled_residual, atol, coupled_rel_res, rtol)
        Iter += 1
    return {}
