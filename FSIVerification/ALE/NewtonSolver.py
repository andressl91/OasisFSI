from dolfin import *
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, assemble, solve, \
norm, MPI, mpi_comm_world, PETScPreconditioner, PETScKrylovSolver

def Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res,VVQ):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VVQ)
    Jac = derivative(F, udp,dw)                # Jacobi

    while rel_res > rtol and residual > atol and Iter < max_it:
        if Iter == 0 or Iter == 10:
            A = assemble(Jac)
            A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, udp.vector()) for bc in bcs]

        #solve(A, udp_res.vector(), b, "superlu_dist")
        list_linear_solver_methods()
        solve(A, udp_res.vector(), b, "mumps")

        udp.vector()[:] = udp.vector()[:] + lmbda*udp_res.vector()[:]
        #udp.vector().axpy(1., udp_res.vector())
        [bc.apply(udp.vector()) for bc in bcs]
        rel_res = norm(udp_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    return udp
