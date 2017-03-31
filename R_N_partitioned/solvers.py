def Newton_manual_s(F, d, bc_d, atol, rtol, max_it, lmbda,d_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(V2)
    #F_1 = assemble(F) + Mass_s_b_L
    Jac_1 = derivative(F, d,dw)                # Jacobi

    #a = assemble(Jac_1)#,keep_diagonal)
    #a.vector().zero()
    while rel_res > rtol and residual > atol and Iter < max_it:
         A = assemble(Jac_1,keep_diagonal=True)#, tensor = a)
         #A.ident_zeros()
         b = assemble(-F)

         [bc.apply(A, b, d.vector()) for bc in bc_d]

         #solve(A, udp_res.vector(), b, "superlu_dist")

         solve(A, d_res.vector(), b)#, "mumps")

         d.vector()[:] = d.vector()[:] + lmbda*d_res.vector()[:]
         #udp.vector().axpy(1., udp_res.vector())
         [bc.apply(d.vector()) for bc in bc_d]
         rel_res = norm(d_res, 'l2')
         residual = b.norm('l2')

         if MPI.rank(mpi_comm_world()) == 0:
             print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
         % (Iter, residual, atol, rel_res, rtol)
         Iter += 1

    return d


def Newton_manual(F, up, bcs, atol, rtol, max_it, lmbda,udp_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VQ)
    F = assemble(F)
    Jac = derivative(F, up,dw)   # Jacobi

    a = assemble(Jac)
    #a.vector().zero()
    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac, tensor = a)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, up.vector()) for bc in bc_u]

        #solve(A, udp_res.vector(), b, "superlu_dist")

        solve(A, up_res.vector(), b)#, "mumps")

        up.vector()[:] = up.vector()[:] + lmbda*up_res.vector()[:]
        #udp.vector().axpy(1., udp_res.vector())
        [bc.apply(up.vector()) for bc in bc_u]
        rel_res = norm(up_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    return up
