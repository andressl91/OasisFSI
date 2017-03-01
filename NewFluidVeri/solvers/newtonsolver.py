from dolfin import *

def Newton_manual(F, VQ, u_, p_, up_, inlet, bcs, T, dt):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VQ)
    Jac = derivative(F, up_["n"], dw)
    atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;

    t = 0
    while t < T:

        if t < 2:
            inlet.t = t;
        if t >= 2:
            inlet.t = 2;

        while rel_res > rtol and residual > atol and Iter < max_it:
            A = assemble(Jac, keep_diagonal = True)
            A.ident_zeros()
            b = assemble(-F)

            [bc.apply(A, b, up_["n"].vector()) for bc in bcs]

            #solve(A, udp_res.vector(), b, "superlu_dist")

            solve(A, udp_res.vector(), b)#, "mumps")

            up_["n"].vector()[:] = up_["n"].vector()[:] + lmbda*udp_res.vector()[:]
            #udp.vector().axpy(1., udp_res.vector())
            [bc.apply(up_["n"].vector()) for bc in bcs]
            rel_res = norm(udp_res, 'l2')
            residual = b.norm('l2')

            if MPI.rank(mpi_comm_world()) == 0:
                print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
            % (Iter, residual, atol, rel_res, rtol)
            Iter += 1
