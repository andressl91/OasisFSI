from dolfin import *
import numpy as np
import sys, shutil, os
#sys.path.append('/Users/Andreas/Desktop/OasisFSI/FSIVerification/semi-imp-project/Fluid_verification/tabulate')
sys.path.append('../Fluid_verification/tabulate')
from tabulate import tabulate

#default values
T = 1.0; dt = []; rho  = 10.0; mu = 1.0; N = []; L = 1;
solver = "chorin"; save_step = 1; fig = False; con = True

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

def NS(N, dt, T, L, rho, mu, solver):
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

    # Define function spaces (P2-P1)
    V1 = VectorFunctionSpace(mesh, "CG", 3, constrained_domain = test)
    V = VectorFunctionSpace(mesh, "CG", 3, constrained_domain = test)
    Q = FunctionSpace(mesh, "CG", 2, constrained_domain = test)
    W = V*Q

    # Define trial and test functions
    up = TrialFunction(W)
    u, p = split(up)
    vq = TestFunction(W)
    v, q = split(vq)

    u_hat = TrialFunction(V1)
    psi = TestFunction(V1)

    # Define boundary conditions
    bcu = []
    bcp = []

    # Create functions
    u0 = project(u_e, V, solver_type="bicgstab")
    p0 = project(p_e, Q)
    u1 = Function(V)
    up1 = Function(W)

    # Define coefficients
    k = Constant(dt)
    #f = Constant((0, 0, 0))
    nu = Constant(mu/rho)

    def eps(u):
        return 0.5*(grad(u) + grad(u).T)

    # Advection-diffusion step (explicit coupling)
    F1 = (1./k)*inner(u_hat - u0, psi)*dx + inner(grad(u_hat)*u0, psi)*dx + \
         2.*nu*inner(eps(u_hat), eps(psi))*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Projection step(implicit coupling)
    F2 = (rho/k)*inner(u - u1, v)*dx - inner(p, div(v))*dx + inner(div(u), q)*dx
    a2 = lhs(F2)
    L2 = rhs(F2)

    # Assemble matrices
    A1 = assemble(a1); A2 = assemble(a2)
    b1 = None; b2 = None
    #list_krylov_solver_preconditioners()

    pc = PETScPreconditioner("jacobi")
    sol = PETScKrylovSolver("bicgstab", pc)

    t = 0
    count = 0;
    if MPI.rank(mpi_comm_world()) == 0:
        print "Computing for N = %g, t = %g" % (N, dt)
    while t < T:
        #if MPI.rank(mpi_comm_world()) == 0:
            #print "Iterating for time %f" % t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1, tensor=b1)
        sol.solve(A1, u1.vector(), b1)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2, tensor=b2)
        #solve(A2, up1.vector(), b2, "gmres", "hypre_amg")
        #solve(lhs(F2) == rhs(F2), up1, bcu)
        solve(A2, up1.vector(), b2)
        end()

        u_, p_ = up1.split(True)

        count += 1
        u0.assign(u_)
        p0.assign(p_)
        t += dt


    time.append(toc())
    p_e.t = t
    u_e.t = t

    u_e = interpolate(u_e, V)
    u0 = interpolate(u0, V)


    #axpy adds multiple of given vector
    uen = norm(u_e.vector())
    u_e.vector().axpy(-1, u0.vector())
    final_error = norm(u_e.vector())/uen
    E.append(final_error)
    #1.12288485362 ,0.994325155573, 0.998055223955, 1.00105884625

    #ue_array = u_e.vector().array()
    #u0_array = u0.vector().array()
    #E.append(np.abs(ue_array - u0_array).max() )
    #1.04675917577 ,0.993725964394,0.995776032497, 0.993133459716

    #L2_u= errornorm(u_e, u0, norm_type='l2', degree_rise=3)
    #E.append(L2_u);

    #1.05268498797, 0.994425378923,0.998542651789,1.00301762883

    h.append(mesh.hmin())
    #degree = V.dim() #DOF Degrees of freedom


set_log_active(False)

time = []; E = []; h = []

nu = mu/rho

if MPI.rank(mpi_comm_world()) == 0:
    print "SOLVING FOR METHOD %s Reynolds number %.2f\n" % (solver, (2./nu))

#dt = [0.1, 0.05, 0.01]#, 0.005, 0.001]
#N = [8, 16, 32]

for t in dt:
    for n in N:
        NS(n, t, T, L, rho, mu, solver)



if len(dt) == 1:
    check = N; opp = dt
else:
    check = dt; opp = N

    #check = N if len(Time) is 1 if Time is len(N) is 1 else 0

    #for i in range(len(E)-1):
        #print np.log(E[i+1]/E[i])/np.log(check[i+1]/check[i])
if fig == True:
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title("Log plot of E and h")
    plt.loglog(check, E)
    plt.show()

if MPI.rank(mpi_comm_world()) == 0:
    print
    print "#################################### - L2 NORM - ####################################\n"
    table = []
    headers = ['N']
    for i in range(len(N)):
        li = []
        li.append(str(N[i]))
        for t in range(len(dt)):
            li.append("%e" % E[len(N)*t + i])
            li.append("%e" % time[len(N)*t + i])
        table.append(li)
    for i in range(len(dt)):
        headers.append("dt = %.g" % dt[i])
        headers.append("Runtime")
    print tabulate(table, headers)
    #print tabulate(table, headers, tablefmt="fancy_grid")


    if con == True:
        print
        print "############################### - CONVERGENCE RATE - ###############################\n"

        table = []
        headers = ['N']
        #for i in range(len(N)):
        for n in range(len(opp)):
            li = []
            li.append(str(opp[n]))
            for i in range(len(check)-1):
                #conv = np.log(E[i+1]/E[i])/np.log(check[i+1]/check[i])
                E_1 =  E[(i+1)*len(opp) + n]; E_0 =  E[i*len(opp) + n]
                conv = np.log(E_1/E_0)/np.log(check[i+1]/check[i])

                li.append(conv)
            table.append(li)
        for i in range(len(check)-1):
            headers.append("%g to %g" % (check[i], check[i+1]))
        print tabulate(table, headers)
        #print tabulate(table, headers, tablefmt="fancy_grid")
