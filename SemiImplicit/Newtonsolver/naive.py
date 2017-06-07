from dolfin import *

def solver_setup(F_correction, F_solid_linear, \
                 F_solid_nonlinear, DVP, dvp_, **monolithic):

    F_solid = F_solid_linear + F_solid_nonlinear

    chi = TrialFunction(DVP)
    Jac_correction = derivative(F_correction, dvp_["n"], chi)
    Jac_solid = derivative(F_solid, dvp_["n"], chi)

    return dict(Jac_correction=Jac_correction, Jac_solid=Jac_solid, F_solid=F_solid)

def Fluid_extrapolation(F_extrapolate, DVP, dvp_, bcs_w, mesh_file, \
                        w_f, k, **semimp_namespace):

    a = lhs(F_extrapolate)
    L = rhs(F_extrapolate)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs_w]

    print "Solving fluid extrapolation"
    solve(A, w_f.vector(), b)

    #Update deformation in mixedspace
    d_n1 = dvp_["n-1"].sub(1, deepcopy=True)
    d = DVP.sub(0).dofmap().collapse(mesh_file)[1].values()
    dvp_["tilde"].vector()[d] = d_n1.vector() + float(k)*w_f.vector()
    dvp_["n"].vector()[d]     = d_n1.vector() + float(k)*w_f.vector()
    
    return dict(w_f=w_f, dvp_=dvp_)

def Fluid_tentative(F_tentative, DVP, V, dvp_, v_tilde, bcs_tent, \
    v_tilde_n1, mesh_file, **semimp_namespace):


    a = lhs(F_tentative)
    L = rhs(F_tentative)
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs_tent]

    print "Solving tentative velocity"
    solve(A, v_tilde.vector(), b)

    #Update tentative n-1 vector
    v_tilde_n1.vector().zero()
    v_tilde_n1.vector().axpy(1, v_tilde.vector())

    #Convert solution into mixedspacefunction
    v = DVP.sub(1).dofmap().collapse(mesh_file)[1].values()
    dvp_["tilde"].vector()[v] = v_tilde.vector()
    #assign(dvp_["n"].sub(0), dvp_["n-1"].sub(0))
    return dict(v_tilde_n1=v_tilde_n1, v_tilde=v_tilde, dvp_=dvp_)


def Fluid_correction(F_correction, Jac_correction, bcs_corr, \
                dvp_, fluid_sol, dvp_res, rtol, atol, max_it, T, t, **monolithic):

    Iter      = 0
    fluid_residual   = 1
    fluid_rel_res    = fluid_residual
    lmbda = 1
    print "Fluid correction\n"
    while fluid_rel_res > rtol and fluid_residual > atol and Iter < max_it:
        if Iter % 6  == 0:# or (last_rel_res < rel_res and last_residual < residual):
            print "assebmling new JAC"
            A = assemble(Jac_correction, keep_diagonal = True)
            #A.axpy(1.0, A_pre, True)
            A.ident_zeros()
            [bc.apply(A) for bc in bcs_corr]
            fluid_sol.set_operator(A)
        #A = assemble(Jac_correction, keep_diagonal=True)
        #A.ident_zeros()

        b = assemble(-F_correction)
        [bc.apply(b, dvp_["n"].vector()) for bc in bcs_corr]
        #[bc.apply(A, b, dvp_["n"].vector()) for bc in bcs]

        fluid_sol.solve(dvp_res.vector(), b)
        #fluid_sol.solve(A, dvp_res.vector(), b)
        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs_corr]
        fluid_rel_res = norm(dvp_res, 'l2')
        fluid_residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, fluid_residual, atol, fluid_rel_res, rtol)
        Iter += 1

    return dict(fluid_rel_res=fluid_rel_res, fluid_residual=fluid_residual ,t=t)

def Solid_momentum(F_solid, Jac_solid, bcs_solid, \
                dvp_, solid_sol, dvp_res, rtol, atol, max_it, T, t, **monolithic):

    Iter      = 0
    solid_residual   = 1
    solid_rel_res    = solid_residual
    lmbda = 1
    print "Solid momentum\n"
    while solid_rel_res > rtol and solid_residual > atol and Iter < max_it:
        if Iter % 6  == 0:# or (last_rel_res < rel_res and last_residual < residual):
            print "assebmling new JAC"
            A = assemble(Jac_solid, keep_diagonal = True)
            #A.axpy(1.0, A_pre, True)
            A.ident_zeros()
            [bc.apply(A) for bc in bcs_solid]
            solid_sol.set_operator(A)
        #A = assemble(Jac_solid, keep_diagonal=True)
        #A.ident_zeros()

        b = assemble(-F_solid)

        [bc.apply(b, dvp_["n"].vector()) for bc in bcs_solid]
        #[bc.apply(A, b, dvp_["n"].vector()) for bc in bcs]

        solid_sol.solve(dvp_res.vector(), b)
        #solid_sol.solve(A, dvp_res.vector(), b)
        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs_solid]
        solid_rel_res = norm(dvp_res, 'l2')
        solid_residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, solid_residual, atol, solid_rel_res, rtol)
        Iter += 1

    return dict(t=t, solid_rel_res=solid_rel_res, solid_residual=solid_residual)
