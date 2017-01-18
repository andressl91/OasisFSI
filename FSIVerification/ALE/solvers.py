from dolfin import *
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, assemble, solve, \
norm, MPI, mpi_comm_world, PETScPreconditioner, PETScKrylovSolver
def integrateFluidStress(p, u):
  eps   = 0.5*(grad(u) + grad(u).T)
  sig   = -p*Identity(2) + 2.0*mu_f*eps
  sig1 = J_(u)*sig*inv(F_(u)).T
  traction  = dot(sig1, -n)

  forceX = traction[0]*ds(5) + traction[0]*ds(6)
  forceY = traction[1]*ds(5) + traction[1]*ds(6)
  fX = assemble(forceX)
  fY = assemble(forceY)

  return fX, fY
def Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res,VVQ):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VVQ)
    Jac = derivative(F, udp,dw)                # Jacobi

    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, udp.vector()) for bc in bcs]

        #solve(A, udp_res.vector(), b, "superlu_dist")

        solve(A, udp_res.vector(), b)#, "mumps")

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
