from dolfin import *

def ALE_projection_fluid_solver(F1, F2, u_, p_, up_, u_tilde, u0_tilde, \
                            bcu, bcs, T, dt, u_e, p_e, w_e, d_e, t_):

    dis_x = []; dis_y = []; time = []

    a1 = lhs(F1); L1 = rhs(F1)

    a2 = lhs(F2); L2 = rhs(F2)

    t = dt
    # TODO: Find stable iterator to achive good convergence, LU works now
    #pc = PETScPreconditioner("jacobi")
    #sol = PETScKrylovSolver("bicgstab", pc)

    while t <= T:

        # FIXME: Change to assemble seperatly and try different solver methods
        # (convex problem as long as we do not have bulking)
        u_e.t = t
        p_e.t = t
        t_.assign(t)

        begin("Computing tentative velocity")
        #b1 = assemble(L1, tensor=b1)
        #[bc.apply(A1, b1) for bc in bcu]
        #sol.solve(A1, u_tilde.vector(), b1)
        solve(a1 == L1, u_tilde, bcu)
        u0_tilde.assign(u_tilde)
        end()

        # Pressure correction and projection on divergencefree field
        begin("Computing pressure correction")
        solve(a2 == L2, up_["n"], bcs)
        #b2 = assemble(L2, tensor=b2)
        #[bc.apply(A2, b2) for bc in bcs]
        #solve(A2, up.vector(), b2)
        end()


        # Update solution
        times = ["n-2", "n-1", "n"]
        for i, t_tmp in enumerate(times[:-1]):
            up_[t_tmp].vector().zero()
            up_[t_tmp].vector().axpy(1, up_[times[i+1]].vector())

        time.append(t)
        up_["n-1"].assign(up_["n"])

        t += dt
        #if MPI.rank(mpi_comm_world()) == 0:
            #print "Time: ",t

    u_e.t = t - dt
    p_e.t = t - dt

    u_sol, p_sol = up_["n"].split(True)

    p_error = errornorm(p_e, p_sol, norm_type='L2', degree_rise = 3)
    u_error = errornorm(u_e, u_sol, norm_type='L2', degree_rise = 3)

    return u_error, p_error, time
