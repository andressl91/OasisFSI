from dolfin import *

def solver_setup(F_correction, F_solid_linear, fluid_sol, solid_sol, \
                 F_solid_nonlinear, DW, dw_, **monolithic):

    F_solid = F_solid_linear + F_solid_nonlinear

    a = lhs(F_correction)
    A_corr = assemble(a, keep_diagonal=True)
    #A_corr.ident_zeros()
    fluid_sol.set_operator(A_corr)

    chi = TrialFunction(DW)
    Jac_solid = derivative(F_solid, dw_["n"], chi)
    solid_sol.parameters['reuse_factorization'] = True


    return dict(A_corr=A_corr, Jac_solid=Jac_solid, fluid_sol=fluid_sol, F_solid=F_solid)

def Fluid_extrapolation(F_extrapolate, DW, dw_, vp_, bcs_w, mesh_file, \
                        k, **semimp_namespace):

    a = lhs(F_extrapolate)
    L = rhs(F_extrapolate)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs_w]

    u_sol = Function(DW)
    print "Solving fluid extrapolation"
    solve(A, dw_["tilde"].vector(), b)

    #d = DVP.sub(0).dofmap().collapse(mesh_file)[1].values()
    #dvp_["n"].vector()[d] = w_f.vector()

    return dict(dw_=dw_)

def Fluid_tentative(F_tentative, VP, vp_, bcs_tent, \
     mesh_file, **semimp_namespace):


    a = lhs(F_tentative)
    L = rhs(F_tentative)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs_tent]

    print "Solving tentative velocity"
    solve(A, vp_["tilde"].vector(), b)


    return dict(vp_=vp_)


def Fluid_correction(mesh_file, VP, A_corr, F_correction, bcs_corr, \
                vp_, dw_, fluid_sol, T, t, **monolithic):

    a = lhs(F_correction)
    A_corr = assemble(a, keep_diagonal=True)
    A_corr.ident_zeros()
    b = assemble(rhs(F_correction))
    [bc.apply(A_corr, b) for bc in bcs_corr]


    print "Solving correction velocity"
    solve(A_corr, vp_["n"].vector(), b)

    print "deformation norm", norm(dw_["n"].sub(0, deepcopy=True), "l2")
    return dict(t=t, vp_=vp_)


def Solid_momentum(F_solid, Jac_solid, bcs_solid, vp_, \
                DW, dw_, solid_sol, dw_res, rtol, atol, max_it, T, t, **monolithic):


    print "Pressure norm", norm(vp_["n"].sub(1, deepcopy=True), "l2")
    Iter      = 0
    solid_residual   = 1
    solid_rel_res    = solid_residual
    lmbda = 1
    print "Solid momentum\n"

    while solid_rel_res > rtol and solid_residual > atol and Iter < max_it:

        #A, b = assemble_system(Jac_solid, -F_solid, bcs_solid, keep_diagonal=True)
        #A.ident_zeros()
        #[bc.apply(dw_["n"].vector()) for bc in bcs_solid]
        #solve(A, dw_res.vector(), b)

        if Iter % 6  == 0:# or (last_rel_res < rel_res and last_residual < residual):
            print "assebmling new JAC"
            A = assemble(Jac_solid, keep_diagonal = True)
            [bc.apply(A) for bc in bcs_solid]
            A.ident_zeros()
            solid_sol.set_operator(A)

        #A = assemble(Jac_solid, keep_diagonal=True)
        #A.ident_zeros()

        b = assemble(-F_solid)

        [bc.apply(b, dw_["n"].vector()) for bc in bcs_solid]
        solid_sol.solve(dw_res.vector(), b)

        dw_["n"].vector().axpy(lmbda, dw_res.vector())
        [bc.apply(dw_["n"].vector()) for bc in bcs_solid]
        solid_rel_res = norm(dw_res, 'l2')
        solid_residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.8e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, solid_residual, atol, solid_rel_res, rtol)
        Iter += 1
    print "NEWO"
    print "Pressure norm", norm(vp_["n"].sub(1, deepcopy=True), "l2")
    print "deformation norm", norm(dw_["n"].sub(0, deepcopy=True), "l2")
    return dict(t=t, dw_=dw_, \
    solid_rel_res=solid_rel_res, solid_residual=solid_residual)
