from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#default values
T = 1.0; dt = []; rho  = 10.0; mu = 1.0; v_deg = 1; p_deg = 1
solver = "Newton"; fig = False;

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-T":
        T = float(sys.argv[1]); del sys.argv[1]
    elif option == "-dt":
        dt.append(float(sys.argv[1])); del sys.argv[1]
    elif option == "-v_deg":
        v_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-p_deg":
        p_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-rho":
        rho = Constant(float(sys.argv[1])); del sys.argv[1]
    elif option == "-mu":
        mu = float(sys.argv[1]); del sys.argv[1]
    elif option == "-solver":
        solver = str(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    elif option.isdigit() == False:
        dt.append(float(option)); #del sys.argv[1]
    else:
        print sys.argv[0], ': invalid option', option

if len(dt) == 0:
    dt = [0.1]

def fluid(mesh_file, T, dt, solver, fig, v_deg, p_deg):
    if mesh_file == None:
        mesh = Mesh("course.xml")
    mesh = Mesh(mesh_file)
    #plot(mesh)
    #interactive()

    #plot(mesh,interactive=True)
    V = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
    Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

    U_dof = V.dim()
    mesh_cells = mesh.num_cells()

    VQ = MixedFunctionSpace([V,Q])

    # BOUNDARIES
    H = 0.41

    Inlet  = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2.2))
    Walls = AutoSubDomain(lambda x: "on_boundary" and ( near(x[1], 0) or near(x[1], 0.41) ))
    Cyl = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0]>1e-6 and x[0]<1 and x[1] < 3*H/4 and x[1] > H/4)

    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    Cyl.mark(boundaries, 1) #Circle gets overwritten, need separate nos for integration
    Inlet.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)
    Walls.mark(boundaries, 4)

    ds = Measure("ds", subdomain_data = boundaries)
    n = FacetNormal(mesh)
    plot(boundaries,interactive=True)

    #BOUNDARY CONDITIONS

    Um = 0.3
    H = 0.41
    D = 0.1

    #STEADY FLOW
    inlet = Expression(("4*Um*x[1]*(H - x[1]) / pow(H, 2)"\
    ,"0"), Um = Um, H = H)

    u_inlet  = DirichletBC(VQ.sub(0), inlet,    boundaries, 2)
    nos_circ = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    p_out    = DirichletBC(VQ.sub(1), 0, boundaries, 3)

    bcs = [u_inlet, nos_wall, nos_circ]


    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)
    u ,p = TrialFunctions(VQ)

    u0 = Function(V)
    u1 = Function(V)

    #Physical parameter
    k = Constant(dt)
    t = 0.0

    rho = 1.0
    nu = 0.001
    mu = nu*rho

    def sigma_fluid(p,u):
        return -p*Identity(2) + mu * (grad(u) + grad(u).T)
    #MY WAY
    def integrateFluidStress(p, u):

        eps   = 0.5*(grad(u) + grad(u).T)
        sig   = -p*Identity(2) + 2.0*mu*eps

        traction  = dot(sig, -n)

        forceX  = traction[0]*ds(1)
        forceY  = traction[1]*ds(1)
        fX      = assemble(forceX)
        fY      = assemble(forceY)

        return fX, fY

    #MEK4300 WAY
    def iintegrateFluidStress(p, u):
    	n = -FacetNormal(mesh)
    	n1 = as_vector((1.0,0)) ; n2 = as_vector((0,1.0))
    	nx = dot(n,n1) ; ny = dot(n,n2)
    	nt = as_vector((ny,-nx))

    	#mf = FacetFunction("size_t", mesh)
    	#mf.set_all(0)
    	#Cyl.mark(mf, 1)
    	#dS = ds[mf]

        ut = dot(nt,u_)
    	Uav = (2*Um)/3
    	Fd = assemble((rho*nu*dot(grad(ut),n)*ny-p_*nx)*ds(1))
    	Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx+p_*ny)*ds(1))
    	#Cd = (2*Fd)/(rho*Uav**2*D)
    	#Cl = (2*Fl)/(rho*Uav**2*D)

        return Fd, Fl


    #OASIS WAY
    def iintegrateFluidStress(p, u):
        D = 0.1
        tau = -p*Identity(2) + mu*(grad(u)+grad(u).T)
        R = VectorFunctionSpace(mesh, 'R', 0)
        c = TestFunction(R)
        ff = FacetFunction("size_t", mesh, 0)
        Cyl.mark(ff, 1)
        df = ds[ff]
        U_mean = Um*2./3
        #forces = assemble(dot(dot(tau, -n), c)*df(1)).array()*2/U_mean**2/D
        #print "Cd = {}, CL = {}".format(*forces)
        forces = assemble(dot(dot(tau, -n), c)*ds(1))

        return forces[0], forces[1]


    Re = 2/3*Um*D/nu
    print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
    print "DOF = %f,  cells = %f" % (U_dof, mesh_cells)


    if solver == "Newton":
        up = Function(VQ)
        u, p = split(up)

        # Fluid variational form
        F = (rho/k)*inner(u - u0, phi)*dx \
            + rho*inner(dot(u, grad(u)), phi) * dx \
            + inner(sigma_fluid(p,u), grad(phi))*dx - inner(div(u),eta)*dx

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations \nComputing for t = %g" % ( dt)

        while t <= T:
            time.append(t)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time t = %.3f" % t

            if t < 2:
                inlet.t = t;
            if t >= 2:
                inlet.t = 2;

            J = derivative(F, up)
            #solve(F == 0, up, bcu, J=J)

            problem = NonlinearVariationalProblem(F, up, bcs, J)
            solver  = NonlinearVariationalSolver(problem)

            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-6
            prm['newton_solver']['relative_tolerance'] = 1E-6
            prm['newton_solver']['maximum_iterations'] = 6
            prm['newton_solver']['relaxation_parameter'] = 1.0


            solver.solve()

            u_, p_ = up.split(True)
            u0.assign(u_)

            drag, lift =integrateFluidStress(p_, u_)

            c_d = 2*drag/(rho*Um*Um*0.1)
            c_l = 2*lift/(rho*Um*Um*0.1)
            if MPI.rank(mpi_comm_world()) == 0:
                print  "Time: ", t ,"C_d: ",c_d, "    C_l: ",c_l

        	Drag.append(drag)
        	Lift.append(lift)

            t += dt

    if solver == "Piccard":

        u, p = TrialFunctions(VQ)
        up = Function(VQ)

        F = (rho/k)*inner(u - u1, phi)*dx \
            + rho*inner(dot(u0, grad(u)), phi) * dx \
            + inner(sigma_fluid(p,u), grad(phi))*dx - inner(div(u),eta)*dx

        count = 0;
        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Piccard iterations \nt = %g" % (dt)

        while t <= T:
            time.append(t)

            if t < 2:
                inlet.t = t;
            if t >= 2:
                inlet.t = 2;

            eps = 10
            k_iter = 0
            max_iter = 20
            while eps > 1E-6 and k_iter < max_iter:
                solve(lhs(F) == rhs(F), up, bcs)

                u_, p_ = up.split(True)
                eps = errornorm(u_,u0,degree_rise=3)
                k_iter += 1
                u0.assign(u_)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time t = %.3f" % t
                print "iterations: %d  error: %.3e" %(k_iter, eps)

            drag, lift =integrateFluidStress(u_, p_)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t ," drag: ",drag, "lift: ",lift
        	Drag.append(drag)
        	Lift.append(lift)

            u1.assign(u_)
            t += dt

    if solver == "Steady":

        up = Function(VQ)
        u, p = split(up)

        # Fluid variational form
        F =  rho*inner(dot(u, grad(u)), phi) * dx \
        + inner(sigma_fluid(p, u), grad(phi))*dx - inner(div(u),eta)*dx

        #F OASIS
        F =  inner(dot(u, grad(u)), phi) * dx \
        + nu*inner(grad(u), grad(phi))*dx - inner(p, div(phi))*dx - inner(eta, div(u))*dx

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations \n STEADY_STATE"

        J = derivative(F, up)

        problem = NonlinearVariationalProblem(F, up, bcs, J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-10
        prm['newton_solver']['relative_tolerance'] = 1E-10
        prm['newton_solver']['maximum_iterations'] = 7
        prm['newton_solver']['relaxation_parameter'] = 1.0


        solver.solve()

        u_, p_ = up.split(True)

        u_file = File("velocity.pvd")
        u_file << u_

        p_file = File("pressure.pvd")
        p_file << p_

        drag, lift = integrateFluidStress(p_, u_)

        U_m = 2./3.*Um

        c_d = 2.*drag/(rho*U_m*U_m*D)
        c_l = 2.*lift/(rho*U_m*U_m*D)
        if MPI.rank(mpi_comm_world()) == 0:
            print  "C_d: ",c_d, "    C_l: ",c_l


    if fig == True:
        if MPI.rank(mpi_comm_world()) == 0:
            plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
            plt.xlabel("Time Seconds")
            plt.ylabel("Lift force Newton")
            plt.plot(time, Lift, label='dt  %g' % dt)
            plt.legend(loc=4)



count = 1;
for m in ["course.xml"]:#, "cylinder.xml"]:
    if MPI.rank(mpi_comm_world()) == 0:
        plt.figure(count)
    for t in dt:
        Drag = []; Lift = []; time = []
        fluid(m, T, t, solver, fig, v_deg, p_deg)
    count += 1;


if fig == True:
    if MPI.rank(mpi_comm_world()) == 0:
        plt.show()
