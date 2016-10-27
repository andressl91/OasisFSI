from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

print mesh
#default values
T = 1.0; dt = []; v_deg = 1; p_deg = 1
solver = "Newton"; steady = False; fig = True;

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
    elif option == "-solver":
        solver = str(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = True
        #fig = bool(sys.argv[1]); del sys.argv[1]
    elif option.isdigit() == False:
        dt.append(float(option)); #del sys.argv[1]
    else:
        print sys.argv[0], ': invalid option', option

if len(dt) == 0:
    dt = [0.1]

H = 0.41
D = 0.1
R = D/2.
Um = 2.0 #CASE 3
nu = 0.001
rho = 1000.
mu = rho*nu

def fluid(mesh, T, dt, solver, steady, fig, v_deg, p_deg):

    #plot(mesh,interactive=True)
    V = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
    Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

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


    ##UNSTEADY FLOW
    inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
    ,"0"), t = 0.0, Um = Um, H = H)

    inlet_steady = Expression(("1.5*Um*x[1]*(H - x[1]) / (pow((H/2.0), 2)) "\
    ,"0"), Um = Um, H = H)


    u_inlet = DirichletBC(VQ.sub(0), inlet, boundaries, 2)
    nos_geo = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    u_inlet0 = DirichletBC(VQ.sub(0), inlet, boundaries, 2)
    nos_geo0 = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall0 = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    p_out = DirichletBC(VQ.sub(1), 0, boundaries, 3)

    bcs = [u_inlet, nos_geo, nos_wall]
    bcs0 = [u_inlet0, nos_geo0, nos_wall0]


    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)
    u ,p = TrialFunctions(VQ)

    u0 = Function(V)
    u1 = Function(V)

    k = Constant(dt)
    t = 0.0


    #MEK4300 WAY
    def FluidStress(p, u):
      n = -FacetNormal(mesh)
      n1 = as_vector((1.0,0)) ; n2 = as_vector((0,1.0))
      nx = dot(n,n1) ; ny = dot(n,n2)
      nt = as_vector((ny,-nx))

      ut = dot(nt, u)
      Fd = assemble((rho*nu*dot(grad(ut),n)*ny - p*nx)*ds(1))
      Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx + p*ny)*ds(1))

      return Fd, Fl

    def sigma_f(p, u):
        return - p*Identity(2) + mu*(grad(u) + grad(u).T)

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

    Re = Um*D*rho/mu
    if MPI.rank(mpi_comm_world()) == 0:
        print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
        print "DOF = %f,  cells = %f" % (U_dof, mesh_cells)


    if solver == "Newton":
    	up = Function(VQ)
    	u, p = split(up)

        up0 = Function(VQ)
    	u0, p0 = split(up)

        theta = 1;
    	# Fluid variational form

        F = ( rho/k*inner(u - u0, phi) \
            + rho*(theta*inner(dot(grad(u), u), phi) + (1 - theta)*inner(dot(grad(u0), u0), phi) ) \
            + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0) , grad(phi)) ) *dx \
            - eta*div(u)*dx


    	if MPI.rank(mpi_comm_world()) == 0:
    		print "Starting Newton iterations \nComputing for t = %g" % ( dt)
        vel_file = File("velocity/velocity.pvd")
    	while t <= T:
    		time.append(t)

    		if t < 2:
    			inlet.t = t;
    		if t >= 2:
    			inlet = inlet_steady;

    		J = derivative(F, up)

    		problem = NonlinearVariationalProblem(F, up, bcs, J)
    		solver  = NonlinearVariationalSolver(problem)

    		prm = solver.parameters
    		prm['newton_solver']['absolute_tolerance'] = 1E-10
    		prm['newton_solver']['relative_tolerance'] = 1E-10
    		prm['newton_solver']['maximum_iterations'] = 10
    		prm['newton_solver']['relaxation_parameter'] = 1.0


    		solver.solve()

    		u_, p_ = up.split(True)
                #vel_file << u_
    		up0.assign(up)


    		drag, lift =integrateFluidStress(p_, u_)
    		if MPI.rank(mpi_comm_world()) == 0:
    		  print "Time: ",t ," drag: ",drag, "lift: ",lift
    		Drag.append(drag)
    		Lift.append(lift)

    		t += dt

    if solver == "Newton2":

        up = Function(VQ)
        u, p = split(up)

        up0 = Function(VQ)
        u0, p0 = split(up0)

        theta = 1.0

        F = (rho*theta*inner(dot(grad(u), u), phi) + rho*(1 - theta)*inner(dot(grad(u0), u0), phi)   \
        + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi) ) )*dx   \
        + eta*div(u)*dx

        dw = TrialFunction(VQ)

        atol, rtol = 1e-7, 1e-7                  # abs/rel tolerances
        lmbda      = 1.0                            # relaxation parameter
        WD_inc      = Function(VQ)                  # residual
        Iter      = 0                               # number of iterations
        residual   = 1                              # residual (To initiate)
        rel_res    = residual                       # relative residual
        max_it    = 100                              # max iterations
        bcs_u = []

        for bc in bcs:
            bc.apply(up.vector())

        for i in bcs0:
            i.homogenize()
            bcs_u.append(i)

        if MPI.rank(mpi_comm_world()) == 0:
    		print "Starting Newton iterations"
        #vel_file = File("velocity/velocity.pvd")
        while t <= T:
            time.append(t)
            if t < 2:
                inlet.t = t;
            if t >= 2:
                inlet.t = 2;

            dF_W = derivative(F, up, dw)

            while rel_res > rtol and residual > atol and Iter < max_it:
                A, b = assemble_system(dF_W, -F, bcs_u)

                # Must be implemented in FSI #############
                #A.ident_zeros()                         #
                #[bc.apply(A, b) for bc in bcs_u]        #
                ##########################################
                solve(A, WD_inc.vector(), b)

                rel_res = norm(WD_inc, 'l2')

                a = assemble(F)
                for bc in bcs_u:
                    bc.apply(a)
                residual = b.norm('l2')

                up.vector()[:] += lmbda*WD_inc.vector()
                if MPI.rank(mpi_comm_world()) == 0:
                    print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
                % (Iter, residual, atol, rel_res, rtol)
                Iter += 1

            #print type(d)
            for bc in bcs:
                bc.apply(up.vector())

            u_, p_  = up.split(True)

            #file_v = File("velocity.pvd")
            #file_v << u_

            #file_p = File("pressure.pvd")
            #file_p << p_
            #Reset counters
            Iter      = 0
            residual   = 1
            rel_res    = residual

            U_m = 2./3.*Um
            drag, lift =integrateFluidStress(p_, u_)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t ," drag: ",drag, "lift: ",lift
            Drag.append(drag)
            Lift.append(lift)
            up0.assign(up)

            t += dt

    if solver == "Piccard":

        u, p = TrialFunctions(VQ)
        up = Function(VQ)

        F = (rho/k)*inner(u - u1, phi)*dx +\
        	rho*inner(grad(u)*u0, phi)*dx + \
        	mu*inner(grad(u), grad(phi))*dx - \
        	div(phi)*p*dx - eta*div(u)*dx


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
        		print "iterations: %d  error: %.3e" %(k_iter, eps)

        	drag, lift =integrateFluidStress(p_, u_)
        	if MPI.rank(mpi_comm_world()) == 0:
        		print "Time: ",t ," drag: ",drag, "lift: ",lift
        	Drag.append(drag)
        	Lift.append(lift)

        	u1.assign(u_)
        	t += dt
    if MPI.rank(mpi_comm_world()) == 0:
        print "Max Lift Force %.4f" % max(Lift)
        print "Max Drag Force %.4f" % max(Drag)
        print "Min Lift Force %.4f" % max(Lift)
        print "Min Drag Force %.4f" % max(Drag)


        print "Mean Lift force %.4f" % (0.5*(max(Lift) + min(Lift) ))
        print "Mean Drag force %.4f" % (0.5*(max(Drag) + min(Drag) ))

        print "Lift amplitude %.4f" % (0.5*(max(Lift) - min(Lift) ))
        print "Drag amplitude %.4f" % (0.5*(max(Drag) - min(Drag) ))


    if fig == True:
        if MPI.rank(mpi_comm_world()) == 0:
            plt.figure(1)
            plt.title("LIFT CFD3 \n Re = %.1f, dofs = %d, cells = %d \n T = %g, dt = %g"
            % (Re, U_dof, mesh_cells, T, dt) )
            plt.xlabel("Time Seconds")
            plt.ylabel("Lift force Newton")
            plt.plot(time, Lift, label='dt  %g' % dt)
            plt.legend(loc=4)
            plt.savefig("lift.png")

            plt.figure(2)
            plt.title("LIFT CFD3\n Re = %.1f, dofs = %d, cells = %d \n T = %g, dt = %g"
            % (Re, U_dof, mesh_cells, T, dt) )
            plt.xlabel("Time Seconds")
            plt.ylabel("Drag force Newton")
            plt.plot(time, Drag, label='dt  %g' % dt)
            plt.legend(loc=4)
            plt.savefig("drag.png")
            #plt.show()




for m in ["turek1.xml"]:
    mesh = Mesh(m)
    #mesh = refine(mesh)
    for t in dt:
        Drag = []; Lift = []; time = []
        fluid(mesh, T, t, solver, steady, fig, v_deg, p_deg)
#if MPI.rank(mpi_comm_world()) == 0:
#    np.savetxt("Lift.txt", Lift, delimiter=',')
#    np.savetxt("Drag.txt", Drag, delimiter=',')
#    np.savetxt("time.txt", time, delimiter=',')
