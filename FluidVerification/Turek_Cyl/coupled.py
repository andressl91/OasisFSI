from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#default values
v_deg = 1; p_deg = 1
solver = "Newton"; fig = False;

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-v_deg":
        v_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-p_deg":
        p_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-solver":
        solver = str(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    else:
        print sys.argv[0], ': invalid option', option

H = 0.41
D = 0.1
R = D/2.
Um = 0.3
nu = 0.001
rho = 1
mu = rho*nu


def fluid(mesh, solver, fig, v_deg, p_deg):
  #plot(mesh)
  #interactive()

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
  
  ug, pg = TrialFunctions(VQ)
  phig, etag = TestFunctions(VQ)

  u0 = Function(V)


  #Physical parameter
  t = 0.0

  def sigma_fluid(p,u):
	return -p*Identity(2) + mu * (grad(u) + grad(u).T)
 
  #MEK4300 WAY
  def FluidStress(p, u):
	print "MEK4300 WAY"
	n = -FacetNormal(mesh)
	n1 = as_vector((1.0,0)) ; n2 = as_vector((0,1.0))
	nx = dot(n,n1) ; ny = dot(n,n2)
	nt = as_vector((ny,-nx))

	ut = dot(nt, u)
	Fd = assemble((rho*nu*dot(grad(ut),n)*ny - p*nx)*ds(1))
	Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx + p*ny)*ds(1))

	return Fd, Fl


   #MY WAY
  def integrateFluidStress(p, u):
	print "MY WAY!"

	eps   = 0.5*(grad(u) + grad(u).T)
	sig   = -p*Identity(2) + 2.0*mu*eps

	traction  = dot(sig, -n)

	forceX  = traction[0]*ds(1)
	forceY  = traction[1]*ds(1)
	fX      = assemble(forceX)
	fY      = assemble(forceY)

	return fX, fY

  Re = 2./3*Um*D/nu
  print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
  
  if solver == "Newton":
	up = Function(VQ)
	u, p = split(up)

	# Fluid variational form
	F = mu*inner(grad(u), grad(phi))*dx + inner(grad(u)*u, phi)*dx \
	- div(phi)*p*dx - eta*div(u)*dx

	if MPI.rank(mpi_comm_world()) == 0:
	  print "Starting Newton iterations"

	J = derivative(F, up)

	problem = NonlinearVariationalProblem(F, up, bcs, J)
	solver  = NonlinearVariationalSolver(problem)

	prm = solver.parameters
	prm['newton_solver']['absolute_tolerance'] = 1E-19
	prm['newton_solver']['relative_tolerance'] = 1E-10
	prm['newton_solver']['maximum_iterations'] = 20
	prm['newton_solver']['relaxation_parameter'] = 1.0


	solver.solve()
	u_ , p_ = up.split(True)

		
  if solver == "Piccard":

	up = Function(VQ)
	
	if MPI.rank(mpi_comm_world()) == 0:
	  print "Starting Piccard iterations"
	eps = 10
	k_iter = 0
	max_iter = 10

	while eps > 1E-6 and k_iter < max_iter:

	  #SGIMA WRITTEN OUT
	  F = mu*inner(grad(u), grad(phi))*dx + inner(grad(u)*u0, phi)*dx \
	  - div(phi)*p*dx - eta*div(u)*dx
	
	  solve(lhs(F) == rhs(F), up, bcs)
	  u_ , p_ = up.split(True)
	  eps = errornorm(u_, u0, degree_rise=3)
	  u0.assign(u_)

	  k_iter += 1
	  
	  if MPI.rank(mpi_comm_world()) == 0:
		print "iterations: %d  error: %.3e" %(k_iter, eps)
  
	u_ , p_ = up.split(True)
	#u_ , p_ = split(up)

	

  drag, lift = integrateFluidStress(p_, u_)

  U_m = 2./3.*Um

  Cd = 2.*drag/(rho*U_m*U_m*D)
  Cl = 2.*lift/(rho*U_m*U_m*D)

  press = p_.compute_vertex_values()
  dp = press[5]-press[7]

  u,v = u_.split(True)
  uvals = u.compute_vertex_values()
  xmax = 0
  for j in range(len(uvals)):
	if uvals[j] < 0:
		if mesh.coordinates()[j][0] > xmax:
			xmax = mesh.coordinates()[j][0]
  La = xmax - 0.25
  print('U_Dof= %d, cells = %d, Cd = %f, Cl = %f, dp = %f La = %f' \
  % (V.dim(), mesh.num_cells(), Cd, Cl, dp, La))

set_log_active(False)
for m in ["course.xml"]:
  mesh = Mesh(m)
  print "SOLVING FOR MESH %s" % m
  for i in range(2):
		if i > 0:
		  mesh = refine(mesh)
		Drag = []; Lift = []; time = []
		fluid(mesh, solver, fig, v_deg, p_deg)



