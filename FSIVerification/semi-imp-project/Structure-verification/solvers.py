# FIXME: Merge the two solver functions as only solver call is different
def solver_linear(G, d_, w_, wd_, bcs, T, dt):
    dis_x = []; dis_y = []; time = []
    a = lhs(G); L = rhs(G)
    t = 0
    # solver_solid = create_solver("gmres", "hypre_euclid")
    while t <= T:
        # FIXME: Change to assemble seperatly and try different solver methods
        # (convex problem as long as we do not have bulking)
        solve(a == L, wd_["n"], bcs)

        # Update solution
        times = ["n-3", "n-2", "n-1", "n"]
        for i, t_tmp in enumerate(times[:-1]):
            wd_[t_tmp].vector().zero()
            wd_[t_tmp].vector().axpy(1, wd_[times[i+1]].vector())

        # Get displacement
        # TODO: Change to fenicsprobes as this will fail in parallel
        dis_x.append(wd_["n"].sub(1)(coord)[0])
        dis_y.append(wd_["n"].sub(1)(coord)[1])
        time.append(t)

        t += dt
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t

    return dis_x, dis_y, time


def solver_nonlinear(G, d_, w_, wd_, bcs, T, dt):
    dis_x = []; dis_y = []; time = []
    solver_parameters = {"newton_solver": \
                          {"relative_tolerance": 1E-8,
                           "absolute_tolerance": 1E-8,
                           "maximum_iterations": 100,
                           "relaxation_parameter": 1.0}}
    t = 0
    while t <= T:
        solve(G == 0, wd_["n"], bcs, solver_parameters=solver_parameters)

        # Update solution
        times = ["n-3", "n-2", "n-1", "n"]
        for i, t_tmp in enumerate(times[:-1]):
            wd_[t_tmp].vector().zero()
            wd_[t_tmp].vector().axpy(1, wd_[times[i+1]].vector())
            w_[t_tmp], d_[t_tmp] = wd_[t_tmp].split(True)

        # Get displacement
        # TODO: Change to fenicsprobes as this will fail in parallel
        dis_x.append(wd_["n"].sub(1)(coord)[0])
        dis_y.append(wd_["n"].sub(1)(coord)[1])
        time.append(t)

        t += dt
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

    return dis_x, dis_y, time
