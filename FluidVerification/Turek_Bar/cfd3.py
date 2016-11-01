from dolfin import *
import sys, os
import numpy as np
import matplotlib.pyplot as plt

import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description="########################################"
"\nImplementation of Turek test case CFD3\n######################################## \n \n"
"-  The program automaticly stores experiment parameters and plots of lift and drag \n"
"   in the experiment folder \n \n"
"-  For details of numerical benchmark go to:\n   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.1689&rep=rep1&type=pdf",\
 formatter_class=RawTextHelpFormatter, \
 epilog="############################################################################\n"
 "Example --> python cfd3.py -T 0.02 -dt 0.01 -v_deg 2 -p_deg 1 -solver Newton\n"
 "Example --> python cfd3.py -solver Newton2  -v_deg 2 -p_deg 1 -r  (Refines mesh one time, -rr for two etc.) \n"
 "############################################################################")
group = parser.add_argument_group('Parameters')
group.add_argument("-T",      type=float, help="Set degree of pressure                     --> Default=1", default=0.02)
group.add_argument("-dt",      type=float, help="Set degree of pressure                     --> Default=1", default=0.01)
group.add_argument("-p_deg",  type=int, help="Set degree of pressure                     --> Default=1", default=1)
group.add_argument("-v_deg",  type=int, help="Set degree of velocity                     --> Default=2", default=2)
group.add_argument("-theta",  type=float, help="Explicit, Implicit, Cranc-Nic (0, 1, 0.5)  --> Default=1", default=1)
group.add_argument("-discr",  help="Write out or keep tensor in variational form --> Default=1", default="keep")
group.add_argument("-r", "--refiner", action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
group2 = parser.add_argument_group('Solvers')
group2.add_argument("-solver", help="Newton   -- Fenics built-in module \n"
"Newton2  -- Manuell implementation\n"
"Piccard  -- Manuell implementation\n"
"Default  --> Newton", default="Newton")

args = parser.parse_args()
T = args.T
dt = args.dt
solver = args.solver
v_deg = args.v_deg
p_deg = args.p_deg
solver = args.solver
theta = args.theta
discr = args.discr
fig = False


H = 0.41
D = 0.1
R = D/2.
Um = 2.0 #CASE 3
nu = 0.001
rho = 1000.
mu = rho*nu

def fluid(mesh, T, dt, solver, fig, v_deg, p_deg, theta, m, discr):

    n = FacetNormal(mesh)
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

    bcs = [u_inlet, nos_geo, nos_wall, p_out]
    bcs0 = [u_inlet0, nos_geo0, nos_wall0]


    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)
    u ,p = TrialFunctions(VQ)

    k = Constant(dt)
    t = 0.0

    Re = Um*D*rho/mu
    if MPI.rank(mpi_comm_world()) == 0:
        print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
        print "DOF = %f,  cells = %f" % (U_dof, mesh_cells)


    if solver == "Newton" or solver == "Newton2":
        phi, eta = TestFunctions(VQ)
        up = Function(VQ)
    	u, p = split(up)
        #For non-theta
        #u0 = Function(V)

        up0 = Function(VQ)
    	u0, p0 = split(up0)

        if discr == "keep":
            F = rho/k*inner(u - u0, phi) *dx \
            + rho*(theta*inner(dot(grad(u), u), phi) + (1 - theta)*inner(dot(grad(u0), u0), phi) )*dx \
            + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi))  *dx \
            - eta*div(u)*dx

        if discr == "split":
    	    F = rho/k*inner(u - u0, phi)*dx \
            + rho*inner(theta*grad(u)*u + (1 -theta)*grad(u0)*u0, phi) *dx \
            + mu*inner(theta*grad(u) + (1-theta)*grad(u0) , grad(phi))*dx  \
    	    - (theta*div(phi)*p + (1 - theta)*div(phi)*p0)*dx\
            - eta*div(u)*dx


    if solver == "Newton":
        tic()

        J = derivative(F, up)

        problem = NonlinearVariationalProblem(F, up, bcs, J)
        sol  = NonlinearVariationalSolver(problem)

        prm = sol.parameters
        #info(prm,True)  #get full info on the parameters
        prm['nonlinear_solver'] = 'newton'
        prm['newton_solver']['absolute_tolerance'] = 1E-10
        prm['newton_solver']['relative_tolerance'] = 1E-10
        prm['newton_solver']['maximum_iterations'] = 100
        prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = 'mumps'

    	if MPI.rank(mpi_comm_world()) == 0:
    		print "Starting Newton iterations \nComputing for t = %g" % ( dt)
        vel_file = File("velocity/velocity.pvd")
    	while t <= T:
    		time.append(t)

    		if t < 2:
    			inlet.t = t;
    		if t >= 2:
    			inlet = inlet_steady;

    		sol.solve()

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
        tic()

        up = Function(VQ)
        u, p = split(up)

        up0 = Function(VQ)
        u0, p0 = split(up0)

        #theta = 1.0

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
        	while eps > 1E-7 and k_iter < max_iter:
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
    run_time = toc()
    print time

    if MPI.rank(mpi_comm_world()) == 0:
        max_drag = max(Drag); min_drag = min(Drag)
        max_lift = max(Lift); min_lift = min(Lift)

        mean_lift = (0.5*(max(Lift) + min(Lift) ))
        mean_drag = (0.5*(max(Drag) + min(Drag) ))

        lift_amp = (0.5*(max(Lift) - min(Lift) ))
        drag_amp = (0.5*(max(Drag) + min(Drag) ))
        print "Max Lift Force %.4f" % max_lift
        print "Max Drag Force %.4f" % max_drag
        print "Min Lift Force %.4f" % min_lift
        print "Min Drag Force %.4f" % min_drag


        print "Mean Lift force %.4f" % mean_lift
        print "Mean Drag force %.4f" % mean_drag

        print "Lift amplitude %.4f" % lift_amp
        print "Drag amplitude %.4f" % drag_amp

        count = 1
        while os.path.exists("./experiments/cfd3/"+str(count)):
            count+= 1

        os.makedirs("./experiments/cfd3/"+str(count))

        print("Creating report file ./experiments/cfd3/"+str(count)+"/report.txt")
        name = "./experiments/cfd3/"+str(count)+"/report.txt"  # Name of text file coerced with +.txt
        f = open(name, 'w')
        f.write("""CFD3 Turek parameters\n
Re = %(Re)g \nmesh = %(m)s\nDOF = %(U_dof)d\nT = %(T)g\ndt = %(dt)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n"""
"""solver = %(solver)s\ntheta_scheme = %(theta).1f\nDiscretization = %(discr)s\n""" % vars())
        f.write("""Runtime = %f \n\n""" % run_time)

        f.write("""Max Lift Force = %(max_lift)g\n
Min Lift Force = %(min_lift)g\n
Max Drag Force = %(max_drag)g\n
Min Drag Force = %(min_drag)g\n
Mean Lift Force = %(mean_lift)g\n
Mean Drag Force = %(mean_drag)g\n
Amplitude Lift Force = %(lift_amp)g\n
Amplitude Drag Force = %(drag_amp)g\n""" %vars())

        f.close()


    if MPI.rank(mpi_comm_world()) == 0:
        np.savetxt("./experiments/cfd3/"+str(count)+"/Lift.txt", Lift, delimiter=',')
        np.savetxt("./experiments/cfd3/"+str(count)+"/Drag.txt", Drag, delimiter=',')
        np.savetxt("./experiments/cfd3/"+str(count)+"/time.txt", time, delimiter=',')

        plt.figure(1)
        plt.title("LIFT CFD3 \n Re = %.1f, dofs = %d, cells = %d \n T = %g, dt = %g"
        % (Re, U_dof, mesh_cells, T, dt) )
        plt.xlabel("Time Seconds")
        plt.ylabel("Lift force Newton")
        plt.plot(time, Lift, label='dt  %g' % dt)
        plt.legend(loc=4)
        plt.savefig("./experiments/cfd3/"+str(count)+"/lift.png")

        plt.figure(2)
        plt.title("DRAG CFD3\n Re = %.1f, dofs = %d, cells = %d \n T = %g, dt = %g"
        % (Re, U_dof, mesh_cells, T, dt) )
        plt.xlabel("Time Seconds")
        plt.ylabel("Drag force Newton")
        plt.plot(time, Drag, label='dt  %g' % dt)
        plt.legend(loc=4)
        plt.savefig("./experiments/cfd3/"+str(count)+"/drag.png")
        #plt.show()

m = "turek1.xml"
mesh = Mesh(m)
Drag = []; Lift = []; time = []
if args.refiner == None:
    fluid(mesh, T, dt, solver, fig, v_deg, p_deg, theta, m, discr)

else:
    for i in range(args.refiner):
        mesh = refine(mesh)
    fluid(mesh, T, dt, solver, fig, v_deg, p_deg, theta, m, discr)
