from dolfin import *
import numpy as np
import sys, shutil, os
#sys.path.append('/uio/hume/student-u86/andressl/Desktop/Sandbox/TG2D/tabulate-0.7.5')

from tabulate import tabulate

#default values
T = 1.0; dt = []; rho  = 10.0; mu = 1.0; N = []; L = 1;
solver = "Newton"; fig = False; con = True

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-T":
        T = float(sys.argv[1]); del sys.argv[1]
    elif option == "-dt":
        dt.append(float(sys.argv[1])); del sys.argv[1]
    elif option == "-rho":
        rho = Constant(float(sys.argv[1])); del sys.argv[1]
    elif option == "-mu":
        mu = float(sys.argv[1]); del sys.argv[1]
    elif option == "-N":
        N.append(int(sys.argv[1])); del sys.argv[1]
    elif option == "-L":
        L = float(sys.argv[1]); del sys.argv[1]
    elif option == "-save_step":
        save_step = float(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    elif option == "-con":
        con = True;  del sys.argv[1]
    elif option == "-solver":
        solver = sys.argv[1]; del sys.argv[1]
    elif option.isdigit() == False:
        dt.append(float(option)); #del sys.argv[1]
    elif option.isdigit() == True:
        N.append(int(option));
    else:
        print sys.argv[0], ': invalid option', option

if len(N) == 0:
    N = [8]
if len(dt) == 0:
    dt = [0.1, 0.01]

def NS(N, dt, T, L, rho, mu, solver, check):
    tic()

    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)

    class PeriodicDomain(SubDomain):

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
            return bool((near(x[0], -1) or near(x[1], -1)) and
                  (not ((near(x[0], -1) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], -1)))) and on_boundary)

        def map(self, x, y):
            if near(x[0], 1) and near(x[1], 1):
                y[0] = x[0] - 2.0
                y[1] = x[1] - 2.0
            elif near(x[0], 1):
                y[0] = x[0] - 2.0
                y[1] = x[1]
            else:
                y[0] = x[0]
                y[1] = x[1] - 2.0

    constrained_domain = PeriodicDomain()
    test = PeriodicDomain()

    nu = Constant(mu/rho)

    # Define time-dependent pressure boundary condition
    p_e = Expression("-0.25*(cos(2*pi*x[0]) + cos(2*pi*x[1]))*exp(-4*t*nu*pi*pi )", nu=nu, t=0.0)
    u_e = Expression(("-cos(pi*x[0])*sin(pi*x[1])*exp(-2*t*nu*pi*pi)",\
                    "cos(pi*x[1])*sin(pi*x[0])*exp(-2*t*nu*pi*pi)"), nu=nu, t=0)

    V = VectorFunctionSpace(mesh, "CG", 1, constrained_domain = test) # Fluid velocity
    Q  = FunctionSpace(mesh, "CG", 1, constrained_domain = test)       # Fluid Pressure

    u_dof.append(V.dim())
    cells.append(mesh.num_cells())

    VQ = MixedFunctionSpace([V,Q])

    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)

    u0 = project(u_e, V)
    u1 = project(u_e, V) #For Piccard Solver

    # Define boundary conditions
    bcu = []
    bcp = []

    k = Constant(dt)
    rho = Constant(rho)
    mu = Constant(mu)
    def sigma_fluid(p,u):
        return -p*Identity(2) + mu * (grad(u) + grad(u).T)#sym(grad(u))


    if solver == "Newton":
        up = Function(VQ)
        u, p = split(up)

        # Fluid variational form
        F = (rho/k)*inner(u - u0, phi)*dx \
            + rho*inner(dot(u, grad(u)), phi) * dx \
            + inner(sigma_fluid(p,u), grad(phi))*dx - inner(div(u),eta)*dx

        t = 0

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations \nComputing for N = %g, t = %g" % (N, dt)

        while t < T:
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time t = %.3f" % t
            J = derivative(F, up)
            #solve(F == 0, up, bcu, J=J)

            problem = NonlinearVariationalProblem(F, up, bcu, J)
            solver  = NonlinearVariationalSolver(problem)

            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-6
            prm['newton_solver']['relative_tolerance'] = 1E-6
            prm['newton_solver']['maximum_iterations'] = 40
            prm['newton_solver']['relaxation_parameter'] = 1.0


            solver.solve()

            u_, p_ = up.split(True)
            u0.assign(u_)

            t += dt

    if solver == "Piccard":

        u, p = TrialFunctions(VQ)
        up = Function(VQ)

        F = (rho/k)*inner(u - u1, phi)*dx \
            + rho*inner(dot(u0, grad(u)), phi) * dx \
            + inner(sigma_fluid(p,u), grad(phi))*dx - inner(div(u),eta)*dx

        t = 0
        count = 0;
        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Piccard iterations \nComputing for N = %g, t = %g" % (N, dt)
        while t < T:
            #b = assemble(L)
            eps = 10
            k_iter = 0
            max_iter = 20
            while eps > 1E-6 and k_iter < max_iter:
                solve(lhs(F) == rhs(F), up, bcu)

                u_, p_ = up.split(True)
                eps = errornorm(u_,u0,degree_rise=3)
                k_iter += 1
                u0.assign(u_)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time t = %.3f" % t
                print "iterations: %d  error: %.3e" %(k_iter, eps)

            u1.assign(u_)
            t += dt


    time.append(toc())
    p_e.t = t
    u_e.t = t

    u_e = interpolate(u_e, V)

    #Oasis way
    #uen = norm(u_e.vector())
    #u_e.vector().axpy(-1, u0.vector())
    #final_error = norm(u_e.vector())/uen
    #E.append(final_error)

    L2_u= errornorm(u_e, u0, norm_type='l2', degree_rise=3)
    E.append(L2_u);

    if check_val == "N":
        h.append(mesh.hmin())

    if check_val == "dt":
        h.append(dt)
    #h.append(dt)
    #degree = V.dim() #DOF Degrees of freedom


set_log_active(False)

time = []; E = []; h = []
u_dof = []; cells = []
#N = [int(10*np.sqrt(2)**i) for i in range(1, 7)]

#print N
#exit(1)

nu = mu/rho

if MPI.rank(mpi_comm_world()) == 0:
    print "SOLVING Reynolds number %.2f\n" % ((2./nu))
if len(dt) < len(N):
    check = N; opp = dt; check_val = "N"
    for t in dt:
        for n in N:
            NS(n, t, T, L, rho, mu, solver, check_val)
if len(dt) >= len(N):
    check = dt; opp = N; check_val = "dt"
    for n in N:
        for t in dt:
            NS(n, t, T, L, rho, mu, solver, check_val)

if MPI.rank(mpi_comm_world()) == 0:

    if fig == True:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.title("Log plot of E and h")
        for i in range(len(opp)):
            y = [E[i*len(check) + j] for j in range(len(check))]
            x = [h[i*len(check) + j] for j in range(len(check))]
            plt.loglog(x, y,marker='o', linestyle='--', label = 'Conv %g' % opp[i])
        plt.legend(loc=2)
        plt.show()


    print
    print "#################################### - L2 NORM - ####################################\n"
    table = []
    headers = ["N" if opp is N else "dt"]
    for i in range(len(opp)):
        li = []
        li.append(str(opp[i]))
        for t in range(len(check)):
            li.append("%e" % E[i*len(check) + t])
            li.append("%e" % time[i*len(check) + t]) #SJEKKKKK!!
        table.append(li)
    for i in range(len(check)):
        headers.append("dt = %.g" % check[i] if check is dt else "N = %g" % check[i])
        headers.append("Runtime")
    print tabulate(table, headers, tablefmt="fancy_grid")

    print
    print "#################################### - Data - ####################################\n"
    table = []
    headers = ["N" if opp is N else "dt"]
    for i in range(len(opp)):
        li = []
        li.append(str(opp[i]))
        for t in range(len(check)):
            li.append("%e" % u_dof[i*len(check) + t])
            li.append("%e" % cells[i*len(check) + t]) #SJEKKKKK!!
        table.append(li)
    for i in range(len(check)):
        headers.append("U_DOF dt = %.g" % check[i] if check is dt else "U_DOF N = %g" % check[i])
        headers.append("CELLS")
    print tabulate(table, headers, tablefmt="fancy_grid")


    if con == True:
        print
        print "############################### - CONVERGENCE RATE - ###############################\n"

        table = []
        headers = ["N" if opp is N else "dt"]
        #for i in range(len(N)):
        print h
        print E
        for n in range(len(opp)):
            li = []
            li.append(str(opp[n]))
            for i in range(len(check)-1):
                #conv = np.log(E[i+1]/E[i])/np.log(check[i+1]/check[i])
                error = E[n*len(check) + (i+1)] / E[n*len(check) + i]
                h_ = h[n*len(check) + (i+1)] / h[n*len(check) + i]
                conv = np.log(error)/np.log(h_) #h is determined in main solve method

                li.append(conv)
            table.append(li)
        for i in range(len(check)-1):
            headers.append("%g to %g" % (check[i], check[i+1]))
        print tabulate(table, headers, tablefmt="fancy_grid")

    #time = []; E = []; h = []
