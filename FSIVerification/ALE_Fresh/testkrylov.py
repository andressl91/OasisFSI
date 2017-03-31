from fenics import *
import numpy as np

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
	 "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()

# Define mesh
set_log_active(False)
E_u = []; E_p = []; h = []
rho = Constant(1)
mu = Constant(1)
nu = Constant(1)

N = [40]
Dt = [1e-5]
T = 1e-4

#N = [20, 24, 28]
#Dt = [1e-6]
#T = 1e-5

#N = [32]
#Dt = [5e-3, 4e-3, 2e-3, 1e-3]
#T = 2e-2
time_list = []
case = "MMS" #Runs MMS or TG2D
for n in N:
    for dt_ in Dt:
        print "Solving for", n, dt_
        if case == "TG2D":
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
            V = VectorFunctionSpace(mesh, "CG", 2)
            Q = FunctionSpace(mesh, "CG", 1)

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


        F_linear = 1./dt * inner(u - u1, v)*dx
        F_linear -= 1./rho * inner(p, div(v))*dx
        F_linear += nu*inner(grad(u), grad(v))*dx
        F_linear -= inner(phi, div(u))*dx

        if case == "MMS":
            F_linear -= inner(f, v)*dx

        F_nonlinear = inner(dot(u, grad(u)), v)*dx

        F = F_linear + F_nonlinear
        chi = TrialFunction(VQ)
        #J_linear    = derivative(F_linear, up, chi)
        #J_nonlinear = derivative(F_nonlinear, up, chi)

        #_pre = assemble(J_linear)
        #A = Matrix(A_pre)
        #b = None
        Jac = derivative(F, up, chi)


        u1_test = project(u_e, V)
        p1_test = project(p_e, Q)
        assign(up1.sub(0), u1_test)
        assign(up1.sub(1), p1_test)

        #Better initial guess
        assign(up.sub(0), u1_test)
        assign(up.sub(1), p1_test)

        upres = Function(VQ)
        t = 0
        rtol = 1e-6; atol = 1e-6; max_it = 20; lmbda = 1

        # Form for use in constructing preconditioner matrix
        tic()
        while t < T:
            t += dt_
            if case == "MMS":
                t_.assign(t)
                u_e.t_ = t
                p_e.t_ = t
            print "Solving for", t
            Iter      = 0
            residual   = 1
            rel_res    = residual

            while rel_res > rtol and residual > atol and Iter < max_it:
                # Assemble system

                # Create Krylov solver and AMG preconditioner
                solver = KrylovSolver(krylov_method)

                # Associate operator (A) and preconditioner matrix (P)
                solver.set_operators(A)

                [bc.apply(A, b, up.vector()) for bc in bcs]
                #solve(A, upres.vector(), b)
                solver.solve(upres.vector(), bb)
                #up_sol.solve(A, upres.vector(), b)
                up.vector().axpy(lmbda, upres.vector())

                [bc.apply(up.vector()) for bc in bcs]

                rel_res = norm(upres, 'l2')
                residual = b.norm('l2')

                if MPI.rank(mpi_comm_world()) == 0:
                    print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
                % (Iter, residual, atol, rel_res, rtol)
                Iter += 1

            #up1.assign(up)

            up1.vector().zero()
            up1.vector().axpy(1, up.vector())

        time_list.append(toc())
        u1, p1 = up1.split(True)
        u_e.t = t
        p_e.t = t
        E_u.append(errornorm(u_e, u1, degree_rise=3))
        E_p.append(errornorm(p_e, p1, degree_rise=3))
        h.append(mesh.hmin())

check = Dt if len(Dt) > 1 else h

print time_list

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
