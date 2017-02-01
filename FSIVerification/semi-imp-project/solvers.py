from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, assemble, solve, \
norm, MPI, mpi_comm_world, PETScPreconditioner, PETScKrylovSolver

def Newton_manual(F, vd, bcs, J, atol, rtol, max_it, lmbda\
                 , vd_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(J, keep_diagonal = True)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, vd.vector()) for bc in bcs]

        #solve(A, vd_res.vector(), b, "superlu_dist")
        #solve(A, vd_res.vector(), b, "mumps")
        solve(A, vd_res.vector(), b)

        vd.vector().axpy(1., vd_res.vector())
        [bc.apply(vd.vector()) for bc in bcs]
        rel_res = norm(vd_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    #Reset
    residual   = 1
    rel_res    = residual
    Iter = 0

    return vd
