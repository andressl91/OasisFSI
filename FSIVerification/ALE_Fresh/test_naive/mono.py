from fenics import *
import numpy as np

# Define mesh
set_log_active(False)
E_u = []; E_p = []; h = []
rho = Constant(1)
mu = Constant(1)
nu = Constant(1)

N = [8, 12, 16]
Dt = [1e-5]
T = 1e-4

N = [32]
Dt = [5e-3, 4e-3, 2e-3, 1e-3]
T = 2e-2

sourceterm = True
for n in N:
    for dt_ in Dt:
        print "Solving for", n, dt_
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
            VQ = MixedFunctionSpace([V, Q])
            # Define time-dependent pressure boundary condition
            p_e = Expression("-0.25*(cos(2*pi*x[0]) + cos(2*pi*x[1]))*exp(-4*t*nu*pi*pi )",nu=nu, t=0.0, degree=4)
            u_e = Expression(("-cos(pi*x[0])*sin(pi*x[1])*exp(-2*t*nu*pi*pi)",\
                            "cos(pi*x[1])*sin(pi*x[0])*exp(-2*t*nu*pi*pi)"), nu=nu, t=0, degree=4)

            bcs = []

        else:
            t_ = Constant(0)
            mesh = UnitSquareMesh(n, n)
            x = SpatialCoordinate(mesh)
            V = VectorFunctionSpace(mesh, "CG", 3)
            Q = FunctionSpace(mesh, "CG", 2)

            VQ = MixedFunctionSpace([V, Q])

            def sigma_f(p_, u_, mu_f):
                return -p_*Identity(2) + 2.*mu_f*sym(grad(u_))

            u_x = "cos(x[0])*sin(x[1])*sin(t_)"
            u_y = "-sin(x[0])*cos(x[1])*sin(t_)"
            p_c = "cos(x[0])*cos(x[1])*sin(t_)"

            p_e = Expression(p_c, nu=nu, t_=0.0, degree=6)
            u_e = Expression((u_x,\
                            u_y), nu=nu, t_=0, degree=6)

            bcs_u = DirichletBC(VQ.sub(0), u_e, "on_boundary")
            bcs_p = DirichletBC(VQ.sub(1), p_e, "on_boundary")
            bcs = [bcs_u, bcs_p]

            exec("u_x = %s" % u_x)
            exec("u_y = %s" % u_y)
            exec("p_c = %s" % p_c)

            u_vec = as_vector([u_x, u_y])
            f = rho*diff(u_vec, t_) + rho*dot(u_vec, grad(u_vec)) \
            - div(sigma_f(p_c, u_vec, mu))



        up = Function(VQ)
        u, p = split(up)
        up1 = Function(VQ)
        u1, p1 = split(up1)
        #up = Function(VQ)

        v, phi = TestFunctions(VQ)

        #dt_ = 1e-4
        dt = Constant(dt_)

        F = 1./dt * inner(u - u1, v)*dx
        F += inner(dot(u, grad(u)), v)*dx
        F -= 1./rho * inner(p, div(v))*dx
        F += nu*inner(grad(u), grad(v))*dx
        F -= inner(phi, div(u))*dx

        if sourceterm == True:
            F -= inner(f, v)*dx

        u1_test = project(u_e, V)
        p1_test = project(p_e, Q)
        assign(up1.sub(0), u1_test)
        assign(up1.sub(1), p1_test)

        #assignerV = FunctionAssigner(VQ.sub(0), V)
        #assignerV.assign(up1.sub(0), u1_test)
        #assignerQ = FunctionAssigner(VQ.sub(1), Q)
        #assignerQ.assign(up1.sub(1), p1_test)

        t = 0
        #solver_parameters = {"newton_solver": {"report_convergence": False}}
        #xdmf = XDMFFile()
        while t < T:
            t += dt_
            if sourceterm == True:
                t_.assign(t)
                u_e.t_ = t
                p_e.t_ = t
            print "Solving for", t
            solve(F==0, up, bcs)#, solver_parameters=solver_parameters)
            #up1.assign(up)
            u_, p_ = up.split(True)
            assign(up1.sub(0), u_)
            assign(up1.sub(1), p_)
            #u1.assign(u_)
            #p1.assign(p_)

        u1, p1 = up1.split(True)
        u_e.t = t
        p_e.t = t
        E_u.append(errornorm(u_e, u1, degree_rise=3))
        E_p.append(errornorm(p_e, p1, degree_rise=3))
        h.append(mesh.hmin())

check = Dt if len(Dt) > 1 else h

print "------------- Velocity ----------------"

for i in E_u:
    print "Errornorm", i

for i in range(len(E_u) - 1):
    r_u = np.log(E_u[i+1]/E_u[i])/np.log(check[i+1]/check[i])
    print "Convergence", r_u

print "------------- Pressure ----------------"

for i in E_p:
    print "Errornorm", i

for i in range(len(E_u) - 1):
    r_p = np.log(E_p[i+1]/E_p[i])/np.log(check[i+1]/check[i])
    print "Convergence", r_p
