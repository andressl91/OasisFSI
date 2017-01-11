from dolfin import *

def eps(u):
    return 0.5*(grad(u) + grad(u).T)

def chorin_temam(dt, rho, mu, u, u1, u0, v, p, p1, q):
    # Define coefficients
    k = Constant(dt)
    #f = Constant((0, 0, 0))
    nu = Constant(mu/rho)

    # Advection-diffusion step (explicit coupling)
    F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u)*u0, v)*dx + \
         2*nu*inner(eps(u), eps(v))*dx #- inner(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Projection step(implicit coupling)
    a2 = (1./k)*inner(u, v)*dx + inner(p, div(v)*dx + inner(div(u), q)*dx
    L2 = (1./k)*inner(u1)*dx

    return a1, L1, a2, L2, a3, L3
