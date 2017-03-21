from fenics import *

def setup(u_, p_, rho_f, nu, mu_f, v, q, k, dx,**Namespace):

    F =  1./k*inner(u_["n"] - u_["n-1"], v)*dx
    F += inner(dot(grad(u_["n"]), u_["n"]), v)*dx
    F += nu*inner(grad(u_["n"]), grad(v))*dx
    F -= 1./rho_f*inner(p_["n"], div(v))*dx
    F -= inner(q, div(u_["n"]))*dx

    return dict(F = F)

def Newton_solver(Jac, F, bcs, up_, up_res, rtol, max_it, atol, \
                  lmbda, **Namespace):
        Iter      = 0
        residual   = 1
        rel_res    = residual
        while rel_res > rtol and residual > atol and Iter < max_it:
            if Iter == 0 or Iter == 10:
                A = assemble(Jac)
                #A.ident_zeros()
            b = assemble(-F)

            [bc.apply(A, b, up_["n"].vector()) for bc in bcs]

            solve(A, up_res.vector(), b)

            up_["n"].vector()[:] = up_["n"].vector()[:] + lmbda*up_res.vector()[:]
            #udp.vector().axpy(1., up_res.vector())
            [bc.apply(up_["n"].vector()) for bc in bcs]
            rel_res = norm(up_res, 'l2')
            residual = b.norm('l2')

            if MPI.rank(mpi_comm_world()) == 0:
                print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
            % (Iter, residual, atol, rel_res, rtol)
            Iter += 1
        Iter      = 0
        residual   = 1
        rel_res    = residual
