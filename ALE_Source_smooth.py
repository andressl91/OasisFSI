from fenics import *
import numpy as np
I = Identity(2)
set_log_active(False)

#import matplotlib.pyplot as plt
#import matplotlib.tri as tri

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot(obj):
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

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

def sigma_f_new2(u, p, d, mu):
	return -J_(d)*dot(inv(F_(d).T), p*Identity(2)) + mu*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def solver(N, dt, T, u_space, p_space):
	mesh = UnitSquareMesh(N, N)
	x = SpatialCoordinate(mesh)
	n = FacetNormal(mesh)

	V = VectorFunctionSpace(mesh, "CG", u_space)
	Q = FunctionSpace(mesh, "CG", p_space)
	D = VectorFunctionSpace(mesh, "CG", 1)

	W = MixedFunctionSpace([V, Q])
	up = Function(W)
	u, p = split(up)

	w = Function(V)
	d = Function(V)

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
	t_step = float(dt)

	mu = 1
	rho = 1

	d_e = Expression(("t*x[1]",
	                  "t*x[0]"
	                 ), degree = 2, t = t_step)

	w_e = Expression(("x[1]",
	                  "x[0]"
	                 ), degree = 2, t = t_step)

	u_e = Expression(("x[1]",
	                  "x[0]"
	                 ), degree = 2, t = 0)

	p_e = Expression("x[0]", degree = 1, t = 0)

	bc_u = DirichletBC(W.sub(0), u_e, "on_boundary")
	bc_p = DirichletBC(W.sub(1), p_e, "on_boundary")
	bcs = [bc_u, bc_p]

	#u0 = interpolate(u_e, V)
	#p0 = interpolate(p_e, Q)
	assign(up0.sub(0), interpolate(u_e, V))
	assign(up0.sub(1), interpolate(p_e, Q))
	assign(up.sub(0), interpolate(u_e, V))
	assign(up.sub(1), interpolate(p_e, Q))

	assign(w, interpolate(w_e, V))
	assign(d, interpolate(d_e, V))

	t_ = Constant(dt)
	d_x = t_*x[1]
	d_y = t_*x[0]
	d_vec = as_vector([d_x, d_y])

	w_x = x[1]
	w_y = x[0]
	w_vec = as_vector([w_x, w_y])

	u_x = x[1]
	u_y = x[0]
	u_vec = as_vector([u_x, u_y])
	p_c = x[0]

	print d_exp.geometric_dimension()
	print mesh.geometry().dim()
	def extrapolation(d_e, w_e):
		bc_d = [DirichletBC(V, d_e, "on_boundary")]
		bc_w = [DirichletBC(V, w_e, "on_boundary")]
		a = inner(grad(chi), grad(delta))*dx
		L = inner(Constant((0, 0)), delta)*dx
		solve(a == L, d_exp, bc_d)
		solve(a == L, w_exp, bc_d)
        return d_exp, w_exp

    d_vec, w_vec = extrapolation(d_e, w_e)

	# Create right hand side f
    f = rho*diff(u_vec, t_) + rho*dot(grad(u_vec), (u_vec-w_vec)) - div(sigma_f(p_c, u_vec, mu))
	#f = J_(d)*rho*diff(u_vec, t_) \
	#+ J_(d)*rho*inv(F_(d))*dot((u_vec-w_vec), grad(u_vec))\
	#- div(J_(d)*sigma_f_new(u_vec, p_c, d, mu)*inv(F_(d)).T)

	F_fluid = rho/k*inner(u - u0, psi)*dx + rho*inner(dot(grad(u), (u - w_vec)), psi)*dx \
	   + inner(sigma_f(p, u, mu), grad(psi))*dx \
	   - inner(f, psi)*dx + inner(div(u), gamma)*dx

	#F_fluid = (rho/k)*inner(J_(d)*(u - u0), psi)*dx
	#F_fluid += rho*inner(J_(d)*inv(F_(d))*dot(u - w, grad(u)), psi)*dx
	#F_fluid += inner(J_(d)*sigma_f_new(u, p, d, mu)*inv(F_(d)).T, grad(psi))*dx
	#F_fluid -= inner(div(J_(d)*dot(sigma_f_new(u, p, d, mu), inv(F_(d)).T)), psi)*dx
	#F_fluid -= inner(f, psi)*dx
	#F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
		#F_fluid += inner(J_(d)*inv(F_(d)).T*-p*Identity(2) \
		# 		+ J_(d)*mu*(dot(grad(u), inv(F_(d))) + dot(inv(F_(d)).T, grad(u).T))*inv(F_(d)).T, grad(psi))*dx

	#F_fluid -= inner(J_(d)*sigma_f_new(u, p, d, mu)*inv(F_(d)).T*n, psi)*ds
	#F_fluid += inner(J_(d)*sigma_f_new(u, p, d, mu)*inv(F_(d)).T, grad(psi))*dx

	L2_u = []
	L2_p = []
	u_diff = Function(V)
	u_file = XDMFFile(mpi_comm_world(), "u_diff_n%d.xdmf" % N)
	p_diff = Function(Q)
	d_move = Function(D)
	p_file = XDMFFile(mpi_comm_world(), "p_diff_n%d.xdmf" % N)
	while t_step <= T:
		u_e.t = t_step
		p_e.t = t_step
		w_e.t = t_step
		d_e.t = t_step
		t_.assign(t_step)
        d_vec, w_vec = extrapolation(d_e, w_e)

		#assign(w, interpolate(w_e, V))
		#assign(d, interpolate(d_e, V))

		J = derivative(F_fluid, up, phi)
		solve(F_fluid == 0, up, bcs, J = J, solver_parameters={"newton_solver": \
		{"relative_tolerance": 1E-9,"absolute_tolerance":1E-9,"maximum_iterations":100,"relaxation_parameter":1.0}})
		up0.assign(up)
		u_, p_ = up.split(True)
		#u0.assign(u_)
		#u_e.t = t_step - dt
		#p_e.t = t_step - dt
        """
        p_e_diff = project(p_e, Q)
		p_diff.vector().zero()
		p_diff.vector().axpy(1, p_e_diff.vector())
		p_diff.vector().axpy(-1, p_.vector())
		#u_diff = u_e - u_
		p_diff.rename("p_diff", "Error in p for each time step")
		p_file.write(p_diff, t_step)

		u_e_diff = project(u_e, V)
		u_diff.vector().zero()
		u_diff.vector().axpy(1, u_e_diff.vector())
		u_diff.vector().axpy(-1, u_.vector())
		#u_diff = u_e - u_
		u_diff.rename("u_diff", "Error in u for each time step")
		u_file.write(u_diff, t_step)
        """
		#plot(p_diff, interactive = True)
		#plot(u_diff, interactive = True)
		#plot(d, mode = "displacement", interactive = True)
		#interactive()
		t_step += dt
		#d_move = project(d_e, D)
		#ALE.move(mesh, d_move)
		#mesh.bounding_box_tree()

	p_e.t = t_step - dt
	u_e.t = t_step - dt

	E_u.append(errornorm(u_e, u_, norm_type="H1", degree_rise = 3))
	E_p.append(errornorm(p_e, p_, norm_type="L2", degree_rise = 3))

	h.append(mesh.hmin())

	#u_e.t = t_step - dt
	#p_e.t = t_step - dt
	#p_diff = p_e - p_
	#plot(p_diff); interactive()
	"""

N = [8, 10, 20]
#N = [6, 8, 10, 12, 18, 20]
#N = [2**i for i in range(2, 6)]
dt = [1E-5]
T = 1E-4
#N = [40]
#dt = [0.005, 0.0025, 0.00125]

u_space = 2; p_space = 1
E_u = [];  E_p = []; h = []

for n in N:
    for t in dt:
        print "Solving for t = %g, N = %d" % (t, n)
        solver(n, t, T, u_space, p_space)

for i in E_u:
    print "Errornorm Velocity L2", i
print

for i in E_p:
    print "Errornorm Pressure L2", i

print

print "Convergence for P%d - P%d" % (u_space, p_space)
check = dt if len(dt) > 1 else h

for i in range(len(E_u) - 1):
	r_u = np.log(E_u[i+1]/E_u[i]) / np.log(check[i+1]/check[i])
	print "Convergence Velocity", r_u

print

for i in range(len(E_p) - 1):
	r_p = np.log(E_p[i+1]/E_p[i]) / np.log(check[i+1]/check[i])
	print "Convergence Pressure", r_p
