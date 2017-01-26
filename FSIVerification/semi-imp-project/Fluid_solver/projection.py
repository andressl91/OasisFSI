from dolfin import *
def eps(u):
    return 0.5*(grad(u) + grad(u).T)

def projection(k, nu, mu, w, u, u_hat, u0, u0_hat, u_hat_sol, v, p, p1, q):
    # Define coefficients

    # Define coefficients
    k = Constant(dt)
    #f = Constant((0, 0, 0))
    nu = Constant(mu/rho)

    # Advection-diffusion step (explicit coupling)
    F1 = (1./k)*inner(u_hat - u0, psi)*dx + inner(grad(u_hat)*(u0 - w), psi)*dx + \
         2.*nu*inner(eps(u_hat), eps(psi))*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Projection step(implicit coupling)
    F2 = (rho/k)*inner(u - u_hat_sol, v)*dx - inner(p, div(v))*dx + inner(div(u), q)*dx
    a2 = lhs(F2)
    L2 = rhs(F2)


    return a1, L1, a2, L2


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
        solve(A2, p1.vector(), b2)
        end()
