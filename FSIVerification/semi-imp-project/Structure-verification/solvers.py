from fenics import assemble, lhs, rhs, solve, PETScPreconditioner, \
                    PETScKrylovSolver, solve, MPI, mpi_comm_world

"""
Solvers

linear for the linearized models

nonlinear for the reference solution
"""

def solver_linear(G, d_, w_, wd_, bcs, T, dt, action=None, **namespace):
    a = lhs(G); L = rhs(G)
    A = assemble(a)
    b = assemble(L)
    t = 0

    u_prec = PETScPreconditioner("hypre_euclid")
    u_sol = PETScKrylovSolver("gmres", u_prec)
    u_sol.prec = u_prec
    u_sol.parameters.update(krylov_solvers)

    # Solver loop
    while t <= T:
        t += dt

        # Assemble
        assemble(a, tensor=A)
        assemble(L, tensor=b)

        # Apply BC
        for bc in bcs: bc.apply(A, b)

        # Solve
        soler_solid.solve(A, wd_["n"].vector(), b)

        # Update solution
        times = ["n-3", "n-2", "n-1", "n"]
        for i, t_tmp in enumerate(times[:-1]):
            wd_[t_tmp].vector().zero()
            wd_[t_tmp].vector().axpy(1, wd_[times[i+1]].vector())

        # Get displacement
        if callable(action):
            action(wd_, t)

        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t


def solver_nonlinear(G, d_, w_, wd_, bcs, T, dt, action=None, **namespace):
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
        if callable(action):
            action(wd_, t)

        t += dt
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ", t
