from fenics import assemble, lhs, rhs, solve, PETScPreconditioner, \
                    PETScKrylovSolver, solve, MPI, mpi_comm_world, \
                    DOLFIN_EPS

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

    d_prec = PETScPreconditioner("default")
    d_solver = PETScKrylovSolver("gmres", d_prec)
    d_solver.prec = d_prec

    # Solver loop
    while t < (T - dt*DOLFIN_EPS):
        t += dt

        # Assemble
        assemble(a, tensor=A)
        assemble(L, tensor=b)

        # Apply BC
        for bc in bcs: bc.apply(A, b)

        # Solve
        d_solver.solve(A, wd_["n"].vector(), b)

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
                           #"krylov_solver": {"monitor_convergence": True},
                           #"linear_solver": {"monitor_convergence": True},
                           "relaxation_parameter": 0.9}}
    t = 0
    while t < (T - dt*DOLFIN_EPS):
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
