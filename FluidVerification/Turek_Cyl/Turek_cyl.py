from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#default values
T = 1.0; dt = []; v_deg = 1; p_deg = 1
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

H = 0.41
D = 0.1
R = D/2.
Um = 1.5
nu = 0.001
rho = 1.
mu = rho*nu

def fluid(mesh_file, T, dt, solver, fig, v_deg, p_deg):
	if mesh_file == None:
		mesh = Mesh("course.xml")
	mesh = Mesh(mesh_file)
	V = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
	Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

	U_dof = V.dim()
	mesh_cells = mesh.num_cells()

	VQ = V*Q

	# BOUNDARIES

	def top(x, on_boundary): return x[1] > (H-DOLFIN_EPS)
	def bottom(x, on_boundary): return x[1] < DOLFIN_EPS
	def Cyl(x, on_boundary): return sqrt((x[0]-0.2)**2 + (x[1]-0.2)**2) < (R		+DOLFIN_EPS)

	inlet = Expression(("4*Um*x[1]*(H - x[1]) / pow(H, 2)"\
	,"0"), Um = Um, H = H)

	#BOUNDARY CONDITIONS
	bc0 = DirichletBC(VQ.sub(0), Constant((0,0)), top)
	bc1 = DirichletBC(VQ.sub(0), Constant((0,0)), bottom)
	bc2 = DirichletBC(VQ.sub(0), Constant((0,0)), Cyl)
	inflow  = DirichletBC(VQ.sub(0), inlet, "x[0] < DOLFIN_EPS")
	bcs = [bc0, bc1, bc2, inflow]

	Circle= AutoSubDomain(Cyl)
	mf = FacetFunction("size_t", mesh)
	mf.set_all(0)
	Circle.mark(mf, 1)
	ds = Measure("ds", subdomain_data = mf)
	n = FacetNormal(mesh)

  
	# TEST TRIAL FUNCTIONS
	phi, eta = TestFunctions(VQ)
	u ,p = TrialFunctions(VQ)

	u0 = Function(V)
	u1 = Function(V)

	#Physical parameter
	k = Constant(dt)
	t = 0.0

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
	def FluidStress(p, u):
		n = -FacetNormal(mesh)
		n1 = as_vector((1.0,0)) ; n2 = as_vector((0,1.0))
		nx = dot(n,n1) ; ny = dot(n,n2)
		nt = as_vector((ny,-nx))

		ut = dot(nt,u_)

		Fd = assemble((rho*nu*dot(grad(ut),n)*ny-p_*nx)*ds(1))
		Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx+p_*ny)*ds(1))

		return Fd, Fl



	Re = 2./3.*Um*D/nu
	print "SOLVING FOR Re = %f" % Re
	print "DOF = %f,  cells = %f" % (U_dof, mesh_cells)


	if solver == "Newton":
		Cl_max = 0; Cd_max = 0
		up = Function(VQ)
		u, p = split(up)

		# Fluid variational form
		F = (rho/k)*inner(u - u0, phi)*dx \
			+ rho*inner(dot(u, grad(u)), phi) * dx \
			+ mu*inner(grad(u), grad(phi))*dx - div(phi)*p*dx - eta*div(u)*dx

		if MPI.rank(mpi_comm_world()) == 0:
			print "Starting Newton iterations \nComputing for t = %g" % ( dt)

		while t <= T:
			time.append(t)
			if MPI.rank(mpi_comm_world()) == 0:
				print "Time t = %.3f" % t
				
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
			
			Cd = 2*drag/(rho*Um*Um*0.1)
			Cl = 2*lift/(rho*Um*Um*0.1)
			
			Cd_max = Cd if abs(Cd) > Cd_max else Cd_max
			Cl_max = Cl if abs(Cl) > Cl_max else Cl_max

			Drag.append(drag)
			Lift.append(lift)

			t += dt

		print('U_Dof= %d, cells = %d, Cd_max = %f, Cl_max = %f' \
		% (V.dim(), mesh.num_cells(), Cd_max, Cl_max))

	if solver == "Piccard":
		up = Function(VQ)

		F = (rho/k)*inner(u - u1, phi)*dx \
			+ rho*inner(dot(u0, grad(u)), phi) * dx \
			+ mu*inner(grad(u), grad(phi))*dx - div(phi)*p*dx - eta*div(u)*dx

		count = 0;
		if MPI.rank(mpi_comm_world()) == 0:
			print "Starting Piccard iterations \nt = %g" % (dt)

		while t <= T:
			time.append(t)

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

			u1.assign(u_)
			drag, lift =integrateFluidStress(u_, p_)
			
			Cd = 2*drag/(rho*Um*Um*0.1)
			Cl = 2*lift/(rho*Um*Um*0.1)
			
			Cd_max = Cd if abs(Cd) > Cd_max else Cd_max
			Cl_max = Cl if abs(Cl) > Cl_max else Cl_max
			
			Drag.append(drag)
			Lift.append(lift)

			t += dt

	if fig == True:
		if MPI.rank(mpi_comm_world()) == 0:
			plt.figure(1)
			plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
			plt.xlabel("Time Seconds")
			plt.ylabel("Lift force Newton")
			plt.plot(time, Lift, label='dt  %g' % dt)
			plt.legend(loc=4)

			plt.figure(2)
			plt.title("Drag \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
			plt.xlabel("Time Seconds")
			plt.ylabel("Drag force Newton")
			plt.plot(time, Drag, label='dt  %g' % dt)
			plt.legend(loc=4)



for m in ["course.xml"]:
    for t in dt:
        Drag = []; Lift = []; time = []
        fluid(m, T, t, solver, fig, v_deg, p_deg)



if fig == True:
    if MPI.rank(mpi_comm_world()) == 0:
        plt.show()
