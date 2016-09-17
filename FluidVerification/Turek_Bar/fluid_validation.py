from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#default values
T = 1.0; dt = []; rho  = 10.0; mu = 1.0; v_deg = 1; p_deg = 1
solver = "Newton"; steady = False; fig = False;

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-T":
        T = float(sys.argv[1]); del sys.argv[1]
    elif option == "-dt":
        dt.append(float(sys.argv[1])); del sys.argv[1]
    elif option == "-v_deg":
        v_deg = float(sys.argv[1]); del sys.argv[1]
    elif option == "-p_deg":
        p_deg = float(sys.argv[1]); del sys.argv[1]
    elif option == "-rho":
        rho = Constant(float(sys.argv[1])); del sys.argv[1]
    elif option == "-mu":
        mu = float(sys.argv[1]); del sys.argv[1]
    elif option == "-solver":
        solver = str(sys.argv[1]); del sys.argv[1]
    elif option == "-steady":
        steady = True
        #steady = bool(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    elif option.isdigit() == False:
        dt.append(float(option)); #del sys.argv[1]
    else:
        print sys.argv[0], ': invalid option', option

if len(dt) == 0:
    dt = [0.1]

def fluid(mesh_file, T, dt, solver, steady, fig, v_deg, p_deg):
    if mesh_file == None:
        mesh = Mesh("course.xml")
    mesh = Mesh(mesh_file)

    #plot(mesh,interactive=True)
    V = VectorFunctionSpace(mesh, "CG", 1) # Fluid velocity
    Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

    U_dof = V.dim()
    mesh_cells = mesh.num_cells()

    VQ = MixedFunctionSpace([V,Q])

    # BOUNDARIES

    Inlet  = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2.5))
    Walls  = AutoSubDomain(lambda x: "on_boundary" and near(x[1],0) or near(x[1], 0.41))

    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    DomainBoundary().mark(boundaries, 1)
    Inlet.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)
    Walls.mark(boundaries, 4)


    ds = Measure("ds", subdomain_data = boundaries)
    n = FacetNormal(mesh)
    #plot(boundaries,interactive=True)

    #BOUNDARY CONDITIONS

    Um = 0.2
    H = 0.41

    ##UNSTEADY FLOW
    inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
    ,"0"), t = 0.0, Um = Um, H = H)
    #STEADY FLOW
    #inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / (H/2.0*H/2.0)"\
    #,"0"), Um = Um, H = H)

    u_inlet = DirichletBC(VQ.sub(0), inlet, boundaries, 2)
    nos_geo = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    p_out = DirichletBC(VQ.sub(1), 0, boundaries, 3)

    bcs = [u_inlet, nos_geo, nos_wall, p_out]


    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)
    u ,p = TrialFunctions(VQ)

    u0 = Function(V)
    u1 = Function(V)


    #dt = 0.001
    k = Constant(dt)
    #T = 10.0
    t = 0.0

    #Physical parameter

    rho = 1000.0
    nu = 0.001
    mu = rho*nu



    def sigma_fluid(p,u):
        return -p*Identity(2) + mu * (grad(u) + grad(u).T)#sym(grad(u))

    def integrateFluidStress(u, p):

        eps   = 0.5*(grad(u) + grad(u).T)
        sig   = -p*Identity(2) + 2.0*mu*eps

        traction  = dot(sig, -n)

        forceX  = traction[0]*ds(1)
        forceY  = traction[1]*ds(1)
        fX      = assemble(forceX)
        fY      = assemble(forceY)

        return fX, fY

    Re = Um*0.1rho/mu
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
            prm['newton_solver']['maximum_iterations'] = 40
            prm['newton_solver']['relaxation_parameter'] = 1.0


            solver.solve()

            u_, p_ = up.split(True)
            u0.assign(u_)

            drag, lift =integrateFluidStress(u_, p_)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t ," drag: ",drag, "lift: ",lift
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

        inlet.t = 2

        # Fluid variational form
        F =  rho*inner(dot(u, grad(u)), phi) * dx \
        + inner(sigma_fluid(p, u), grad(phi))*dx - inner(div(u),eta)*dx

        #F OASIS
        #F =  rho*inner(dot(u, grad(u)), phi) * dx \
        #+ mu*inner(grad(u), grad(phi))*dx - inner(p, div(phi))*dx - inner(eta, div(u))*dx

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations \n STEADY_STATE"

        J = derivative(F, up)

        problem = NonlinearVariationalProblem(F, up, bcs, J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-6
        prm['newton_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['maximum_iterations'] = 50
        prm['newton_solver']['relaxation_parameter'] = 1.0


        solver.solve()

        u_, p_ = up.split(True)
        u0.assign(u_)

        drag, lift =integrateFluidStress(u_, p_)
        if MPI.rank(mpi_comm_world()) == 0:
            print  "drag: ",drag, "lift: ",lift
        Drag.append(drag)
        Lift.append(lift)

    if fig == True:
        if MPI.rank(mpi_comm_world()) == 0:
            plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
            plt.xlabel("Time Seconds")
            plt.ylabel("Lift force Newton")
            plt.plot(time, Lift, label='dt  %g' % dt)
            plt.legend(loc=4)



count = 1;
for m in ["course.xml", "turek1.xml", "turek2.xml"]:
    if MPI.rank(mpi_comm_world()) == 0:
        plt.figure(count)
    for t in dt:
        Drag = []; Lift = []; time = []
        fluid(m, T, t, solver, steady, fig, v_deg, p_deg)
    count += 1;


if fig == True:
    if MPI.rank(mpi_comm_world()) == 0:
        plt.show()
