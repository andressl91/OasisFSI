from dolfin import *
def eps(u):
    return 0.5*(grad(u) + grad(u).T)

def projection(k, nu, mu, u, u_tilde, u0, u0_tilde, u1, v, p, p1, q):
    # Define coefficients

    # Tentative velocity step
    F1 = (1/k)*inner(u_tilde - u0, v)*dx + inner(grad(u_tilde)*(u0_tilde - w), v)*dx + \
         2*nu*inner(grad(eps(u_tilde)), grad(v))*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = -(1./k)*div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = inner(u1, v)*dx - dot(k*grad(p1), v)*dx

    return a1, L1, a2, L2, a3, L3


def fluid_solve(A1, A2, A3, L1, L2, L3, fluid_solver, pressure_solver):
        b1 = None; b2 = None; b3 = None
        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1, tensor=b1)
        fluid_solver.solve(A1, u1.vector(), b1)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2, tensor=b2)
        solve(A2, p1.vector(), b2, "gmres", "hypre_amg")
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3, tensor=b3)
        pc2 = PETScPreconditioner("jacobi")
        sol2 = PETScKrylovSolver("bicgstab", pc2)
        sol2.solve(A3, u1.vector(), b3)
        end()
