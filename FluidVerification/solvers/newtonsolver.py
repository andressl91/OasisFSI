from dolfin import *

def integrateFluidStress(p, u, mu, n, ds):
    eps   = 0.5*(grad(u) + grad(u).T)
    sig   = -p*Identity(2) + 2.0*mu*eps

    traction  = dot(sig, -n)

    forceX  = traction[0]*ds(1)
    forceY  = traction[1]*ds(1)
    fX      = assemble(forceX)
    fY      = assemble(forceY)

    return fX, fY

Lift = []; Drag = []

def Newton_manual(F, VQ, u_, p_, up_, inlet, bcs, T, dt, n, mu, ds):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VQ)
    Jac = derivative(F, up_["n"], dw)
    atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;
    up_res = Function(VQ)

    Lift = [0]
    Drag = [0]
    Time = [0]

    t = 0
    while t < T:
        print "Solving for t = %g" % t

        if t < 2:
            inlet.t = t;
        if t >= 2:
            inlet.t = 2;

        while rel_res > rtol and residual > atol and Iter < max_it:
            A = assemble(Jac)
            A.ident_zeros()
            b = assemble(-F)

            [bc.apply(A, b, up_["n"].vector()) for bc in bcs]

            solve(A, up_res.vector(), b)

            up_["n"].vector()[:] = up_["n"].vector()[:] + lmbda*up_res.vector()[:]
            #udp.vector().axpy(1., up_res.vector())
            [bc.apply(up_["n"].vector()) for bc in bcs]
            rel_res = norm(up_res, 'l2')
            residual = b.norm('l2')

            if MPI.rank(mpi_comm_world()) == 0:
                print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
            % (Iter, residual, atol, rel_res, rtol)
            Iter += 1

        up_["n-1"].assign(up_["n"])
        u_s, p_s = up_["n"].split(True)
        F_x, F_y = integrateFluidStress(p_s, u_s, mu, n, ds)
        print "LIFT = %g    DRAG = %g \n" % (F_x, F_y)
        Lift.append(F_y)
        Drag.append(F_x)
        Time.append(t)

        Iter      = 0
        residual   = 1
        rel_res    = residual
        t += dt

    return Lift, Drag, Time
