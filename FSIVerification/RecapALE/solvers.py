from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, assemble, solve, \
norm, MPI, mpi_comm_world, PETScPreconditioner, PETScKrylovSolver

def Newton_manual(F, udp, bcs, J, atol, rtol, max_it, lmbda\
                 , udp_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(J)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, udp.vector()) for bc in bcs]

        #solve(A, udp_res.vector(), b, "superlu_dist")
        #solve(A, udp_res.vector(), b, "mumps")
        solve(A, udp_res.vector(), b)

        udp.vector().axpy(1., udp_res.vector())
        [bc.apply(udp.vector()) for bc in bcs]
        rel_res = norm(udp_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    #Reset
    residual   = 1
    rel_res    = residual
    Iter = 0

    #return udp
