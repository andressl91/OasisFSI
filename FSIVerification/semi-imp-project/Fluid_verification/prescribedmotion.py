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

    mesh = RectangleMesh(Point(0,0), Point(2, 1), 20, 20, "crossed")

    # Define function spaces (P2-P1)
    V1 = VectorFunctionSpace(mesh, "CG", 2)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # Define trial and test functions
    up = TrialFunction(W)
    u, p = split(up)
    vq = TestFunction(W)
    v, q = split(vq)

    u_hat = TrialFunction(V1)
    psi = TestFunction(V1)

    # Define boundary conditions

    # Fluid velocity conditions
    class U_bc(Expression):
        def init(self,w):
            self.w = w
        def eval(self,value,x):
            #x_value, y_value = self.w.vector()[[x[0], x[1]]]
            value[0], value[1] = self.w(x)
            #value[0] = x_value
            #value[1] = y_value
        def value_shape(self):
            return (2,)

    u_bc = U_bc(degree=2)

    Inlet = AutoSubDomain(lambda x, on_bnd: near(x[0], 0))
    Outlet = AutoSubDomain(lambda x, on_bnd: near(x[0], 2))
    Walls = Inlet = AutoSubDomain(lambda x, on_bnd: near(x[1], 0) and near(x[1], 2))

    fc = FacetFunction('size_t', mesh, 0)
    Inlet.mark(fc, 1)
    Outlet.mark(fc, 2)
    Walls.mark(fc, 3)

    ds = Measure("ds", subdomain_data = fc)
    n = FacetNormal(mesh)

    # Create functions
    d_e = Expression(("sin(t)*(x[0] - 1)",  "0"), t = 0)
    w_e = Expression(("x[0]*cos(t)", "0"), t = 0)
    u_e = Expression(("x[0]*cos(t)", "-cos(t)"), t = 0)
    p_e = Expression("1")

    # Fluid velocity conditions
    u_wall   = DirichletBC(V1, u_bc, fc, 3)
    u_inlet   = DirichletBC(V1, u_e, fc, 1)
    up_wall   = DirichletBC(W.sub(0), u_bc, fc, 3)
    up_inlet   = DirichletBC(W.sub(0), u_e, fc, 1)

    # Mesh velocity conditions
    w_inlet   = DirichletBC(V1, w_e, fc, 1)
    w_outlet  = DirichletBC(V1, ((0.0, 0.0)), fc, 2)

    # Pressure Conditions
    p_out = DirichletBC(W.sub(1), 0, fc, 2)

    #Assemble boundary conditions
    bcs_w = [w_inlet, w_outlet]
    bcs_u = [u_inlet, u_wall]
    bcs_up = [up_inlet, up_wall, p_out]

    d_tilde = Function(V1)
    d0 = project(d_e, V1)

    w = TrialFunction(V1)
    w1 = Function(V1)
    w0 = project(w_e, V1)
    w_1 = project(d_e, V1)

    psieta = TestFunction(W)
    psi, eta = split(psieta)
    up1 = TrialFunction(W)
    u, p = split(up1)
    up0 = Function(W)
    u0, p0 = up0.split(deepcopy = True)
    up_sol = Function(W)

    u_ = TrialFunction(V1)
    phi = TestFunction(V1)
    u0 = project(u_e, V)
    p0 = project(p_e, Q)


    # Define coefficients
    k = Constant(dt)
    #f = Constant((0, 0, 0))
    mu = Constant(10)
    rho = Constant(1)
    nu = Constant(mu/rho)

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

    #F_expo = inner(u_ - d0 - k*(3./2*w0 - 1./2*w_1), phi)*dx
    d_ = k*w1

    #Laplace of deformation d
    F_smooth = k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds)

    F_tent = (rho/k)*inner(J_(d_)*(u_ - u0), phi)*dx \
            + rho*inner(J_(d_)*inv(F_(d_))*grad(u_)*(u0 - w1), phi)*dx  \
            + inner(2.*mu*J_(d_)*eps(u_)*inv(F_(d_)).T, eps(phi))*dx \
            #+ inner(u_ - w1, phi)*ds(1)

    F_press_upt = (rho/k)*inner(J_(d_)*(u - u_tent), psi)*dx \
    - inner(J_(d_)*p*inv(F_(d_)).T, grad(psi))*dx \
    + inner(div(J_(d_)*inv(F_(d_).T)*u), eta)*dx \
    + inner(dot(u, n), eta)*ds(1) \
    #- 1./k*inner(dot(d - d0, n), eta)*ds(1)
    #+ dot(dot(u('-'), n('-')) - 1./k*dot(d('-') - d0('-'), n('-')), psi('-'))*dS(5)

    a1 = lhs(F_tent)
    a2 = lhs(F_press_upt)

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
        print "here"
        #if MPI.rank(mpi_comm_world()) == 0:
            #print "Iterating for time %f" % t
        inlet.t = t

        solve(lhs(F_smooth) == rhs(F_smooth), w1, bcs_w)
        u_bc.init(w1)

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        [bc.apply(A, b) for bc in bcs_u]
        b1 = assemble(L1, tensor=b1)
        solve(A1 , up_sol.vector(), b1)

        end()

        # Pressure correction
        begin("Computing pressure correction")
        [bc.apply(A, b) for bc in bcs_up]
        b2 = assemble(L2, tensor=b2)
        #solve(A2, up1.vector(), b2, "gmres", "hypre_amg")
        #solve(lhs(F2) == rhs(F2), up1, bcu)
        solve(A2, up1.vector(), b2)
        end()

        u_, p_ = up1.split(True)

        count += 1
        u0.assign(u_)

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
