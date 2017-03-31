from dolfin import *

def projection_fluid_solver(F1, F2, F3, V, Q, u_, p_, u_tilde, u_sol, p_sol, u0_tilde, bcu, bcp, T, dt, u_e, p_e, t_, **vars):

    dis_x = []; dis_y = []; time = []

    a1 = lhs(F1); L1 = rhs(F1)
    a2 = lhs(F2); L2 = rhs(F2)
    a3 = lhs(F3); L3 = rhs(F3)

    t = dt
    # TODO: Find stable iterator to achive good convergence, LU works now

    u_file = XDMFFile(mpi_comm_world(), "u_vel_n.xdmf")
    u_diff = Function(V)

    p_file = XDMFFile(mpi_comm_world(), "p_press_n.xdmf")
    p_diff = Function(Q)

    while t <= T:

        # FIXME: Change to assemble seperatly and try different solver methods
        # (convex problem as long as we do not have bulking)
        u_e.t_ = t
        p_e.t_ = t
        t_.assign(t)

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        solve(a1 == L1, u_sol, bcu)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        solve(a2 == L2, p_sol, bcp)
        end()

        # Velocity correction
        begin("Computing velocity correction")
        solve(a3 == L3, u_sol, bcu)
        end()

        u_["n-1"].assign(u_sol)
        p_["n-1"].assign(p_sol)
        t += dt


        #u0_tilde.assign(u_tilde)
        time.append(t)
        """
        p_e_diff = project(p_e, Q)
        p_diff.vector().zero()
        p_diff.vector().axpy(1, p_e_diff.vector())
        p_diff.vector().axpy(-1, p_s.vector())
        p_diff.rename("p_diff", "Error in p for each time step")
        p_file.write(p_diff, t)

        u_e_diff = project(u_e, V)
        u_diff.vector().zero()
        u_diff.vector().axpy(1, u_e_diff.vector())
        u_diff.vector().axpy(-1, u_s.vector())
        u_diff.rename("u_diff", "Error in u for each time step")
        u_file.write(u_diff, t)
        """
    u_e.t = t - dt
    p_e.t = t - dt


    p_error = errornorm(p_e, p_sol, norm_type='L2', degree_rise = 3)
    u_error = errornorm(u_e, u_sol, norm_type='L2', degree_rise = 3)

    return u_error, p_error, time
