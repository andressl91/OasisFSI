from dolfin import *
import numpy as np

E_u = []; E_p = []; h = []
N = [20, 25, 30]
Dt = [1e-7]
T = 1e-6

N = [30]
Dt = [5e-2, 2e-2, 1e-2]
T = 1e-1

mu = 1.; rho = 1.
sourceterm = True
for n in N:
    for dt in Dt:
        print "Solving for", n, dt
        #Parameters
        k = float(dt)
        nu = Constant(mu/rho)

        if sourceterm == False:
            mesh = RectangleMesh(Point(0, 0), Point(2, 2), n, n)

            class PeriodicDomain(SubDomain):
                def inside(self, x, on_boundary):
                    # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
                    return bool((near(x[0], 0) or near(x[1], 0)) and
                          (not ((near(x[0], 0) and near(x[1], 2)) or
                                (near(x[0], 2) and near(x[1], 0)))) and on_boundary)

                def map(self, x, y):
                    if near(x[0], 2) and near(x[1], 2):
                        y[0] = x[0] - 2.0
                        y[1] = x[1] - 2.0
                    elif near(x[0], 2):
                        y[0] = x[0] - 2.0
                        y[1] = x[1]
                    else:
                        y[0] = x[0]
                        y[1] = x[1] - 2.0

            constrained_domain = PeriodicDomain()

            V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=constrained_domain)
            Q = FunctionSpace(mesh, "CG", 1, constrained_domain=constrained_domain)

            # Define time-dependent pressure boundary condition
            p_e = Expression("-0.25*(cos(2*pi*x[0]) + cos(2*pi*x[1]))*exp(-4*t*nu*pi*pi )",nu=nu, t=0.0, degree=4)
            u_e = Expression(("-cos(pi*x[0])*sin(pi*x[1])*exp(-2*t*nu*pi*pi)",\
                            "cos(pi*x[1])*sin(pi*x[0])*exp(-2*t*nu*pi*pi)"), nu=nu, t=0, degree=4)

            bcs_u = []
            bcs_p = []

        else:
            t_ = Constant(0)
            mesh = UnitSquareMesh(n, n)
            x = SpatialCoordinate(mesh)
            V = VectorFunctionSpace(mesh, "CG", 3)
            Q = FunctionSpace(mesh, "CG", 2)

            def sigma_f(p_, u_, mu_f):
                return -p_*Identity(2) + 2.*mu_f*sym(grad(u_))

            u_x = "cos(x[0])*sin(x[1])*sin(t_)"
            u_y = "-sin(x[0])*cos(x[1])*sin(t_)"
            p_c = "cos(x[0]*x[0] + x[1]*x[1])*sin(t_)"

            p_e = Expression(p_c, nu=nu, t_=0.0, degree=6)
            u_e = Expression((u_x,\
                            u_y), nu=nu, t_=0, degree=6)

            bcs_u = [DirichletBC(V, u_e, "on_boundary")]
            bcs_p = [DirichletBC(Q, p_e, "on_boundary")]

            exec("u_x = %s" % u_x)
            exec("u_y = %s" % u_y)
            exec("p_c = %s" % p_c)

            u_vec = as_vector([u_x, u_y])
            f = rho*diff(u_vec, t_) + rho*dot(u_vec, grad(u_vec)) \
            - div(sigma_f(p_c, u_vec, mu))

        u_ = {}; p_ = {};
        time = ["n", "n-1", "sol"]

        for temp in time:
            if temp == "n":
                u_[temp] = TrialFunction(V)
                p_[temp] = TrialFunction(Q)

            else:
                u_[temp] = Function(V)
                p_[temp] = Function(Q)

        v = TestFunction(V)
        q = TestFunction(Q)

        #u_["n-1"].assign(interpolate(u_e, V))
        #p_["n-1"].assign(interpolate(p_e, Q))

        u_["n-1"] = interpolate(u_e, V)
        p_["n-1"] = interpolate(p_e, Q)

        #Tentative Velocity
        F1 = rho/k*inner(u_["n"] - u_["n-1"], v)*dx \
           + rho*inner(dot(u_["n-1"], grad(u_["n"])), v)*dx \
           + mu*inner(grad(u_["n"]), grad(v))*dx
        if sourceterm == True:
            F1 -= inner(f, v)*dx

        a1 = lhs(F1); L1 = rhs(F1)

        F2 = k*dot(grad(p_["n"]), grad(q))*dx + dot(div(u_["sol"]), q)*dx
        a2 = lhs(F2); L2 = rhs(F2)

        F3 = inner(u_["n"] - u_["sol"], v)*dx \
           + k*inner(grad(p_["sol"]), v)*dx

        a3 = lhs(F3); L3 = rhs(F3)

        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)
        b1 = None; b2 = None; b3 = None
        [bc.apply(A1) for bc in bcs_u]
        [bc.apply(A2) for bc in bcs_p]
        #[bc.apply(A3) for bc in bcs_u]

        progress = Progress('Time-stepping')
        set_log_level(PROGRESS)

        t = 0
        while t < T:
            t += dt
            if sourceterm == True:
                t_.assign(t)
                u_e.t_ = t
                p_e.t_ = t

            print "Solving for t = %g" % t
            A1, b1 = assemble_system(a1, L1, bcs_u)
            #b1 = assemble(L1)
            #[bc.apply(b1) for bc in bcs_u]
            solve(A1, u_["sol"].vector(), b1)

            A2, b2 = assemble_system(a2, L2, bcs_p)
            #b2 = assemble(L2)
            #[bc.apply(b2) for bc in bcs_p]
            solve(A2, p_["sol"].vector(), b2)

            A3, b3 = assemble_system(a3, L3, bcs_u)
            #b3 = assemble(L3)
            #[bc.apply(b3) for bc in bcs_u]
            solve(A3, u_["sol"].vector(), b3)

            #u_e.t = t; p_e.t = t
            #u_["n-1"].assign(interpolate(u_e, V))
            #p_["n-1"].assign(interpolate(p_e, Q))

            u_["n-1"].assign(u_["sol"])
            #p_["n-1"].assign(p_["sol"])
            progress.update(t / T)

        u_e.t = t;
        p_e.t = t
        E_u.append(errornorm(u_e, u_["sol"], norm_type="l2", degree_rise=3))
        E_p.append(errornorm(p_e, p_["sol"], norm_type="l2", degree_rise=3))
        h.append(mesh.hmin())

check = Dt if len(Dt) > 1 else h

for i in E_u:
    print "Errornorm velocity = %g" % i
print

for i in range(len(E_u) - 1):
    r = np.log(E_u[i+1]/E_u[i])/np.log(check[i+1]/check[i])
    print "Convergence", r

print

for i in E_p:
    print "Errornorm pressure = %g" % i

print

for i in range(len(E_p) - 1):
    r = np.log(E_p[i+1]/E_p[i])/np.log(check[i+1]/check[i])
    print "Convergence", r
