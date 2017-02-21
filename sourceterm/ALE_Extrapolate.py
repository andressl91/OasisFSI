from fenics import *
import numpy as np
I = Identity(2)
set_log_active(False)
parameters['allow_extrapolation'] = True

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def eps(U):
    return 0.5*(grad(U) + grad(U).T)

def eps_map(U, D):
    return 0.5*(grad(U)*inv(F_(D)) + inv(F_(D)).T*grad(U).T)

def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

def sigma_f_new(u, p, d, mu):
	return -p*Identity(2) + mu*(dot(grad(u), inv(F_(d))) + dot(inv(F_(d)).T, grad(u).T))

def extrapolation(V, d_e, d_exp, w_e, w_exp, chi, delta):
    bc_d = DirichletBC(V, d_e, "on_boundary")
    bc_w = DirichletBC(V, w_e, "on_boundary")
    a = -inner(grad(chi), grad(delta))*dx
    L = inner(Constant((0, 0)), delta)*dx
    solve(a == L, d_exp, bc_d)
    solve(a == L, w_exp, bc_w)
    return d_exp, w_exp

def solver(N, dt, T, u_space, p_space, implementation, sourceterm, extrapol, save_res):

	mesh = UnitSquareMesh(N, N)
	x = SpatialCoordinate(mesh)
	n = FacetNormal(mesh)

	V = VectorFunctionSpace(mesh, "CG", u_space)
	Q = FunctionSpace(mesh, "CG", p_space)
	D = VectorFunctionSpace(mesh, "CG", 1)

	W = MixedFunctionSpace([V, Q])
	up = Function(W)
	u, p = split(up)

	#Extrapolation of d and w
	chi = TrialFunction(V)
	delta = TestFunction(V)
	d_exp = Function(V)
	w_exp = Function(V)

	phi = TrialFunction(W)
	psi, gamma = TestFunctions(W)

	up0 = Function(W)
	u0, p0 = split(up0)

	k = Constant(dt)
	t = float(dt)

	mu = 1
	rho = 1

	d_e = Expression(("cos(x[1])*sin(t)",
	              "cos(x[0])*sin(t)"
	             ), degree = 2, t = t)

	w_e = Expression(("cos(x[1])*cos(t)",
	              "cos(x[0])*cos(t)"
	             ), degree = 2, t = t)

	u_e = Expression(("cos(x[1])*cos(t)",
	              "cos(x[0])*cos(t)"
	             ), degree = 2, t = 0)

	p_e = Expression("cos(x[1])*cos(t)", degree = 1, t = 0)

	bc_u = DirichletBC(W.sub(0), u_e, "on_boundary")
	bc_p = DirichletBC(W.sub(1), p_e, "on_boundary")

	bcs = [bc_u, bc_p]

	assign(up0.sub(0), project(u_e, V))
	assign(up0.sub(1), project(p_e, Q))

	t_ = Constant(dt)
	d_x = cos(x[1])*sin(t)
	d_y = cos(x[0])*sin(t)
	d_vec = as_vector([d_x, d_y])

	w_x = cos(x[1])*cos(t)
	w_y = cos(x[0])*cos(t)
	w_vec = as_vector([w_x, w_y])

	u_x = cos(x[1])*cos(t)
	u_y = cos(x[0])*cos(t)
	u_vec = as_vector([u_x, u_y])

	p_c = cos(x[1])*cos(t)

	# Create right hand side f
	#Sourceterm no-map
	if sourceterm == "nomap":
	    f = rho*diff(u_vec, t_) + rho*dot(grad(u_vec), (u_vec - w_vec)) - div(sigma_f(p_c, u_vec, mu))

	#Sourceterm with mapping
	elif sourceterm == "map":
	    f = J_(d_vec)*rho*diff(u_vec, t_) \
	    + J_(d_vec)*rho*inv(F_(d_vec))*dot((u_vec-w_vec), grad(u_vec))\
	    - div(J_(d_vec)*sigma_f_new(u_vec, p_c, d_vec, mu)*inv(F_(d_vec)).T)

	#Variational Form no map
	if implementation == "nomap":
	    F_fluid = rho/k*inner(u - u0, psi)*dx + rho*inner(dot(grad(u), (u - w_vec)), psi)*dx \
	       + inner(sigma_f(p, u, mu), grad(psi))*dx \
	       - inner(f, psi)*dx(mesh) + inner(div(u), gamma)*dx - inner(Constant(0), gamma)*dx

	#Variational Form map
	if implementation == "map":
	    F_fluid = (rho/k)*inner(J_(d_vec)*(u - u0), psi)*dx
	    F_fluid += rho*inner(J_(d_vec)*inv(F_(d_vec))*dot(u - w_vec, grad(u)), psi)*dx
	    F_fluid += inner(J_(d_vec)*sigma_f_new(u, p, d_vec, mu)*inv(F_(d_vec)).T, eps(psi))*dx
	    F_fluid -= inner(J_(d_vec)*f, psi)*dx
	    F_fluid -= inner(div(J_(d_vec)*inv(F_(d_vec))*u), gamma)*dx

	L2_u = []
	L2_p = []
	h_list = []
	u_vel_file = XDMFFile(mpi_comm_world(), "u_vel_n%d.xdmf" % N)
	u_diff = Function(V)
	u_file = XDMFFile(mpi_comm_world(), "u_diff_n%d.xdmf" % N)

	p_press_file = XDMFFile(mpi_comm_world(), "p_press_n%d.xdmf" % N)
	p_diff = Function(Q)
	d_move = Function(D)
	p_file = XDMFFile(mpi_comm_world(), "p_diff_n%d.xdmf" % N)
	while t <= T:
		u_e.t = t; p_e.t = t
		w_e.t = t; d_e.t = t
		t_.assign(t)
		if extrapol == True:
		    d_vec, w_vec = extrapolation(V, d_e, d_exp, w_e, w_exp, chi, delta)

		J = derivative(F_fluid, up, phi)
		solve(F_fluid == 0, up, bcs, J = J, solver_parameters={"newton_solver": \
		{"relative_tolerance": 1E-9,"absolute_tolerance":1E-9,"maximum_iterations":100,"relaxation_parameter":1.0}})

		up0.assign(up)

		if save_res == True:

		    u_save = project(u_e, V)
		    u_save.rename("u_velocity", "Velocity")
		    u_vel_file.write(u_save, t)

		    p_e_diff = project(p_e, Q)
		    p_diff.vector().zero()
		    p_diff.vector().axpy(1, p_e_diff.vector())
		    p_diff.vector().axpy(-1, p_.vector())
		    #u_diff = u_e - u_
		    p_diff.rename("p_diff", "Error in p for each time step")
		    p_file.write(p_diff, t)

		    u_e_diff = project(u_e, V)
		    u_diff.vector().zero()
		    u_diff.vector().axpy(1, u_e_diff.vector())
		    u_diff.vector().axpy(-1, u_.vector())
		    #u_diff = u_e - u_
		    u_diff.rename("u_diff", "Error in u for each time step")
		    u_file.write(u_diff, t)

		t += dt
		h_list.append(mesh.hmin())

	p_e.t = t - dt
	u_e.t = t - dt
	u_s, p_s = up.split(True)

	E_u.append(errornorm(u_e, u_s, norm_type="L2", degree_rise = 3))
	E_p.append(errornorm(p_e, p_s, norm_type="L2", degree_rise = 3))
	h.append(np.mean(h_list))

#Convergence Time
N = [32]
dt = [0.05/(2**i) for i in range(3)]
T = 0.2

#Convergence Space
#N = [2**i for i in range(1, 5)]
#dt = [2E-6]
#T = 1E-5

E_u = [];  E_p = []; h = []
###################################################

for n in N:
	for t in dt:
		print "Solving for t = %g, N = %d" % (t, n)
		solver(n, t, T,
		u_space = 2,
		p_space = 1, \
		implementation = "map",
		sourceterm = "nomap",
		extrapol = True,
		save_res = False )

###################################################

for i in E_u:
    print "Errornorm Velocity L2", i
print

for i in E_p:
    print "Errornorm Pressure L2", i

print

check = dt if len(dt) > 1 else h

for i in range(len(E_u) - 1):
	r_u = np.log(E_u[i+1]/E_u[i]) / np.log(check[i+1]/check[i])
	print "Convergence Velocity", r_u

print

for i in range(len(E_p) - 1):
	r_p = np.log(E_p[i+1]/E_p[i]) / np.log(check[i+1]/check[i])
	print "Convergence Pressure", r_p
