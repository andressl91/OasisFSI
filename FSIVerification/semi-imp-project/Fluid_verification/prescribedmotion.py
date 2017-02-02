from dolfin import *
import numpy as np
import sys, shutil, os
#sys.path.append('/Users/Andreas/Desktop/OasisFSI/FSIVerification/semi-imp-project/Fluid_verification/tabulate')
#sys.path.append('../Fluid_verification/tabulate')
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

    mesh = RectangleMesh(Point(0,0), Point(2, 1), N, N, "crossed")

    # Define function spaces (P2-P1)
    V1 = VectorFunctionSpace(mesh, "CG", 2)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # Create functions
    d_e = Expression(("sin(t)*(x[0] - 2)",  "0"), t = 0)
    w_e = Expression(("cos(t)*(x[0] - 2)", "0"), t = 0)
    u_e = Expression(("cos(t)*(x[0] - 2)", "-cos(t)*(x[1] - 2)"), t = 0)
    p_e = Expression("0")


    psieta = TestFunction(W)
    psi, eta = split(psieta)
    up = TrialFunction(W)
    u, p = split(up)
    up0 = Function(W)
    u0, p0 = split(up0)
    up_sol = Function(W)

    w = TrialFunction(V1)
    w1 = Function(V1)
    w0 = project(w_e, V1)
    w_1 = project(w_e, V1)

    d_tilde = Function(V1)
    d0 = project(d_e, V1)

    u_ = TrialFunction(V1)
    phi = TestFunction(V1)
    u_tent = Function(V1)
    u0 = project(u_e, V)
    u0_tilde = project(u_e, V)

    # Define coefficients
    k = Constant(dt)
    mu = Constant(1.0)
    rho = Constant(1.0)
    nu = Constant(mu/rho)

    # Define boundary conditions

    # Fluid velocity conditions
    class U_bc(Expression):
        def __init__(self, w):
            self.w = w
        def update(self, w):
            self.w = w
        def eval(self,value,x):
            #x_value, y_value = self.w.vector()[[x[0], x[1]]]
            value[0], value[1] = self.w(x)
            #value[0] = x_value
            #value[1] = y_value
        def value_shape(self):
            return (2,)

    u_bc = U_bc(w1)

    class W_bc(Expression):
        def __init__(self, d_tilde, d0, k):
            self.d_tilde = d_tilde
            self.d0 = d0
            self.k = k
        def eval(self,value,x):
            #x_value, y_value = self.w.vector()[[x[0], x[1]]]
            value[0], value[1] = 1./self.k*(self.d_tilde(x) - self.d0(x))
            #value[0] = x_value
            #value[1] = y_value
        def value_shape(self):
            return (2,)

    w_bc = W_bc(d_tilde, d0, k)


    Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2))
    Walls = Inlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0) or near(x[1], 1)))

    fc = FacetFunction('size_t', mesh, 0)
    Inlet.mark(fc, 1)
    Outlet.mark(fc, 2)
    Walls.mark(fc, 3)

    ds = Measure("ds", subdomain_data = fc)
    n = FacetNormal(mesh)


    # Fluid velocity conditions
    u_wall   = DirichletBC(V1, u_bc, fc, 3)
    u_inlet   = DirichletBC(V1, u_e, fc, 1)
    u_out   = DirichletBC(V1, Constant((0, 0)), fc, 2)

    up_wall   = DirichletBC(W.sub(0), u_bc, fc, 3)
    up_inlet   = DirichletBC(W.sub(0), u_e, fc, 1)
    up_out = DirichletBC(W.sub(0), Constant((0, 0)), fc, 2)

    # Mesh velocity conditions
    w_inlet   = DirichletBC(V1, w_e, fc, 1)
    w_wall   = DirichletBC(V1, u_bc, fc, 3)
    w_outlet  = DirichletBC(V1, ((0.0, 0.0)), fc, 2)

    # Deformation conditions
    d_inlet   = DirichletBC(V1, d_e, fc, 1)
    d_outlet  = DirichletBC(V1, ((0.0, 0.0)), fc, 2)

    # Pressure Conditions
    p_in = DirichletBC(W.sub(1), 0, fc, 1)

    #Assemble boundary conditions
    bcs_w = [w_inlet, w_outlet]
    bcs_d = [d_inlet, d_outlet]
    bcs_u = [u_inlet, u_wall, up_out]
    bcs_up = [up_inlet, up_wall, p_in, up_out]




    I = Identity(2)

    def eps(u):
        return 0.5*(grad(u) + grad(u).T)

    def F_(U):
    	return (I + grad(U))

    def J_(U):
    	return det(F_(U))

    def sigma_f(v,p):
    	return 2*mu*eps(v) - p*Identity(2)

    def sigma_f_hat(v,p,u):
    	return J_(u)*sigma_f(v,p)*inv(F_(u)).T

    d_tilde = Function(V) #Solution vector of F_expo
    d_tilde.vector()[:] = d0.vector()[:] + float(k)*(3./2*w0.vector()[:] - 1./2*w_1.vector()[:])

    ############## Step 1: Definition of new domain

    w_next = Function(V)   #Solution w_n+1 of F_smooth
    d_move = Function(V)   #Def new domain of Lambda_f

    #Laplace of deformation d
    F_smooth = inner(grad(w), grad(phi))*dx
             #- inner(grad(w)*n, phi)*ds

    F_tent = (rho/k)*inner(J_(d_tilde)*(u_ - u0), phi)*dx \
            + rho*inner(J_(d_tilde)*inv(F_(d_tilde))*grad(u_)*(u0_tilde - w1), phi)*dx  \
            + inner(2.*mu*J_(d_tilde)*eps(u_)*inv(F_(d_tilde)).T, eps(phi))*dx \
            + inner(u_ - w1, phi)*ds(1)

    F_press_upt = (rho/k)*inner(J_(d_tilde)*(u - u_tent), psi)*dx \
    - inner(J_(d_tilde)*p*inv(F_(d_tilde)).T, grad(psi))*dx \
    + inner(div(J_(d_tilde)*inv(F_(d_tilde).T)*u), eta)*dx \
    + inner(dot(u, n), eta)*ds(1) \
    - 1./k*inner(dot(d_tilde - d0, n), eta)*ds(1)
    + dot(dot(u, n) - 1./k*dot(d_tilde - d0, n), eta)*ds(1)

    a1 = lhs(F_tent); L1 = rhs(F_tent)
    a2 = lhs(F_press_upt); L2 = rhs(F_press_upt)

    # Assemble matrices
    A1 = assemble(a1); A2 = assemble(a2)
    b1 = None; b2 = None
    #list_krylov_solver_preconditioners()

    pc = PETScPreconditioner("jacobi")
    sol = PETScKrylovSolver("bicgstab", pc)

    t = dt
    count = 0;
    if MPI.rank(mpi_comm_world()) == 0:
        print "Computing for N = %g, t = %g" % (N, dt)
    while t < T:
        d_e.t = t
        u_e.t = t
        w_e.t = t

        #Step 0
        d_tilde.vector()[:] = d0.vector()[:] + float(k)*(3./2*w0.vector()[:] - 1./2*w_1.vector()[:])

        #Step 1
        solve(lhs(F_smooth) == rhs(F_smooth), w1, bcs_w)
        u_bc.update(w1)
        d_tilde.vector()[:] = d0.vector()[:] + float(dt)*w1.vector()[:]


        # Compute tentative velocity step
        begin("Computing tentative velocity")
        A1 = assemble(a1)
        b1 = assemble(L1, tensor=b1)
        [bc.apply(A1, b1) for bc in bcs_u]
        solve(A1 , u_tent.vector(), b1)
        u0_tilde.assign(u_tent)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        A2 = assemble(a2)
        b2 = assemble(L2, tensor=b2)
        [bc.apply(A2, b2) for bc in bcs_up]
        #solve(A2, up1.vector(), b2, "gmres", "hypre_amg")
        #solve(lhs(F2) == rhs(F2), up1, bcu)
        solve(A2, up_sol.vector(), b2)
        end()

        u_, p_ = up_sol.split(True)

        count += 1
        u0.assign(u_)
        d0.assign(d_tilde)
        w_1.assign(w0)
        w0.assign(w1)

        #plot(u0, mode = "displacement")
        t += dt

    #interactive(True)

    time.append(toc())
    u_e.t = t - dt
    w_e.t = t - dt

    u_e = interpolate(u_e, V)
    u0 = interpolate(u0, V)

    #w_e = interpolate(w_e, V)
    #w0 = interpolate(w0, V)

    #axpy adds multiple of given vector
    uen = norm(u_e.vector())
    u_e.vector().axpy(-1, u0.vector())
    final_error = norm(u_e.vector())/uen
    E.append(final_error)
    #1.12288485362 ,0.994325155573, 0.998055223955, 1.00105884625

    #L2_u= errornorm(u_e, u0, norm_type='l2', degree_rise=3)

    h.append(mesh.hmin())



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
